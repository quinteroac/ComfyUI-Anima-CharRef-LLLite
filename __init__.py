import sys

import torch
import torch.nn.functional as F

import comfy.ldm.common_dit
import comfy.patcher_extension
import folder_paths
import nodes
from comfy.ldm.anima.lllite import AnimaLLLitePatch, AnimaLLLiteAttentionPatch
from comfy_api.latest import ComfyExtension, io


def reference_forward_with_controls(ref_adapter, dm, x, timesteps, context, refs, offset_ratio, transformer_options):
    # Reference-frame forward follows ElectronGirl/ComfyUI-Anima-Ref-Adapter (Apache-2.0).
    original_shape = list(x.shape)
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial))
    ref = ref_adapter._pick_ref(refs, tuple(x.shape[-2:])).to(device=x.device, dtype=x.dtype)
    frames = torch.cat([x[:, :, :1], ref[:, :, :1]], dim=2)
    embedded, _, extra = dm.prepare_embedded_sequence(frames, fps=None, padding_mask=None)
    patches = transformer_options.get("patches", {})
    if "post_input" in patches:
        transformer_options = transformer_options.copy()
        transformer_options["model_patch_data"] = {}
        for patch in patches["post_input"]:
            embedded = patch({"img": embedded, "x": x, "transformer_options": transformer_options})["img"]
    _, _, height, width, _ = embedded.shape
    positions = [ref_adapter._frame_positions(0, height, width, 0, 0, x.device),
                 ref_adapter._frame_positions(1, height, width, round(offset_ratio * height), 0, x.device)]
    rope = ref_adapter._rope_from_positions(dm.pos_embedder, *(torch.cat(axis) for axis in zip(*positions)))
    timesteps = timesteps.reshape(x.shape[0], -1)[:, :1]
    timesteps = torch.cat([timesteps, torch.zeros_like(timesteps)], dim=1)
    time_embedding, adaln = dm.t_embedder[1](dm.t_embedder[0](timesteps).to(embedded.dtype))
    time_embedding = dm.t_embedding_norm(time_embedding)
    kwargs = {"rope_emb_L_1_1_D": rope.unsqueeze(1).unsqueeze(0), "adaln_lora_B_T_3D": adaln,
              "extra_per_block_pos_emb": extra, "transformer_options": transformer_options}
    if embedded.dtype == torch.float16:
        embedded = embedded.float()
    for index, block in enumerate(dm.blocks):
        transformer_options["block_index"] = index
        embedded = block(embedded, time_embedding, context, **kwargs)
    out = dm.final_layer(embedded.to(context.dtype), time_embedding, adaln_lora_B_T_3D=adaln)
    out = dm.unpatchify(out)[:, :, :1]
    return out[:, :, :original_shape[-3], :original_shape[-2], :original_shape[-1]]


def asym_target_block(ref_adapter, block, xt, emb, adaln, context, rope, ref_kv, transformer_options):
    # Target stream of the Ref Adapter's _asym_block, with the attn1 and mlp patches that LLLite needs.
    attn = block.self_attn
    sa, ca, mlp = ref_adapter._block_mods(block, emb, adaln)
    residual_dtype, compute_dtype = xt.dtype, emb.dtype
    B, T, H, W, _ = xt.shape
    n = ref_adapter._norm(xt, block.layer_norm_self_attn, sa[1], sa[0]).to(compute_dtype).flatten(1, 3)
    q, k, v = attn.compute_qkv(n, None, rope_emb=rope, transformer_options=transformer_options)
    n_q = q.shape[1]
    k, v = torch.cat([k, ref_kv[0].to(k)], dim=1), torch.cat([v, ref_kv[1].to(v)], dim=1)
    if "compute_attention" in attn.__dict__:
        # Anima Safe PAG expects as many queries as keys, see the Ref Adapter's _asym_block.
        q = torch.cat([q, q.new_zeros(q.shape[0], k.shape[1] - n_q, *q.shape[2:])], dim=1)
    out = attn.compute_attention(q, k, v, transformer_options=transformer_options)[:, :n_q].reshape(B, T, H, W, -1)
    xt = torch.addcmul(xt, sa[2].to(residual_dtype), out.to(residual_dtype))
    n = ref_adapter._norm(xt, block.layer_norm_cross_attn, ca[1], ca[0]).to(compute_dtype).flatten(1, 3)
    out = block.cross_attn(n, context, rope_emb=None, transformer_options=transformer_options).reshape(B, T, H, W, -1)
    xt = torch.addcmul(xt, ca[2].to(residual_dtype), out.to(residual_dtype))
    n = ref_adapter._norm(xt, block.layer_norm_mlp, mlp[1], mlp[0]).to(compute_dtype)
    args = {"x": n, "transformer_options": transformer_options}
    for patch in transformer_options.get("patches", {}).get("mlp_patch", []):
        args = patch(args)
    return torch.addcmul(xt, mlp[2].to(residual_dtype), block.mlp(args["x"]).to(residual_dtype))


def asym_forward_with_controls(ref_adapter, dm, x, timesteps, context, ref_kv, transformer_options):
    # Ref Adapter's _asym_forward (v2 adapters) with the post_input patches run on the target frame.
    original_shape = list(x.shape)
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial))
    sigma = timesteps.reshape(x.shape[0], -1)[:, 0]
    xt, rope, emb, adaln = ref_adapter._frame_embed(dm, x[:, :, :1], 0, 0, 0, sigma)
    patches = transformer_options.get("patches", {})
    if "post_input" in patches:
        transformer_options = transformer_options.copy()
        transformer_options["model_patch_data"] = {}
        for patch in patches["post_input"]:
            xt = patch({"img": xt, "x": x, "transformer_options": transformer_options})["img"]
    batch = xt.shape[0]
    for index, block in enumerate(dm.blocks):
        transformer_options["block_index"] = index
        k, v = ref_kv[index]
        kv = (k.expand(batch, *k.shape[1:]), v.expand(batch, *v.shape[1:])) if k.shape[0] != batch else (k, v)
        xt = asym_target_block(ref_adapter, block, xt, emb, adaln, context, rope, kv, transformer_options)
    out = dm.unpatchify(dm.final_layer(xt.to(context.dtype), emb, adaln_lora_B_T_3D=adaln))[:, :, :1]
    return out[:, :, :original_shape[-3], :original_shape[-2], :original_shape[-1]]


def reference_control_wrapper(ref_adapter, adapter, offset_ratio):
    def wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        refs = [ref for ref in (kwargs.pop(ref_adapter.REF_KEY, None),
                               kwargs.pop(ref_adapter.REF_HALF_KEY, None)) if ref is not None]
        if not refs:
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)
        adapter.install(executor.class_obj)
        if adapter.asym:
            ref_kv = ref_adapter._asym_ref_kv(adapter, executor.class_obj, refs, x, offset_ratio)
        previous = ref_adapter._active["key"]
        ref_adapter._active["key"] = adapter.key
        try:
            if adapter.asym:
                return asym_forward_with_controls(ref_adapter, executor.class_obj, x, timesteps, context,
                                                  ref_kv, kwargs.get("transformer_options", {}))
            return reference_forward_with_controls(ref_adapter, executor.class_obj, x, timesteps, context,
                                                   refs, offset_ratio, kwargs.get("transformer_options", {}))
        finally:
            ref_adapter._active["key"] = previous
    return wrapper


class AnimaRefAdapterControlsLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ExperimentalAnimaRefAdapterControlsLoader",
            display_name="Anima Ref Adapter + Controls (Experimental)",
            category="Anima Ref Adapter/experimental",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.05),
                io.Float.Input("ref_offset_h", default=1.0, min=0.0, max=4.0, step=0.05),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model, lora_name, strength, ref_offset_h):
        loader = nodes.NODE_CLASS_MAPPINGS["AnimaRefAdapterLoader"]
        ref_adapter = sys.modules[loader.__module__]
        patched, = loader().load(model, lora_name, strength, ref_offset_h)
        adapter = ref_adapter._Adapter.get(folder_paths.get_full_path_or_raise("loras", lora_name), strength)
        patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, ref_adapter.WRAPPER_KEY)
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                     ref_adapter.WRAPPER_KEY, reference_control_wrapper(ref_adapter, adapter, ref_offset_h))
        return io.NodeOutput(patched)


class ReferenceLLLiteAttentionPatch(AnimaLLLiteAttentionPatch):
    def __call__(self, q, k, v, pe=None, attn_mask=None, extra_options=None):
        values = {"q": q, "k": k, "v": v}
        control = extra_options["model_patch_data"].get(self.patch)
        if control is None:
            return super().__call__(q, k, v, pe=pe, attn_mask=attn_mask, extra_options=extra_options)
        token_count = control.shape[1]
        target_values = values.copy()
        for name in self.targets:
            flat = values[name].reshape(values[name].shape[0], -1, values[name].shape[-1])
            target_values[name] = flat[:, :token_count]
        result = super().__call__(**target_values, pe=pe, attn_mask=attn_mask, extra_options=extra_options)
        for name in self.targets:
            flat = values[name].reshape(values[name].shape[0], -1, values[name].shape[-1])
            result[name] = torch.cat((result[name], flat[:, token_count:]), dim=1).reshape(values[name].shape)
        return result


class AnimaLLLiteRefApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ExperimentalAnimaLLLiteRefApply",
            display_name="Anima LLLite + Character Ref (Experimental)",
            category="Anima Ref Adapter/experimental",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Custom("MODEL_PATCH").Input("model_patch"),
                io.Image.Input("image"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.05),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model, model_patch, image, strength, start_percent, end_percent):
        sampling = model.get_model_object("model_sampling")
        patch = AnimaLLLitePatch(model_patch, image[..., :3], None, strength,
                                float(sampling.percent_to_sigma(start_percent)),
                                float(sampling.percent_to_sigma(end_percent)))
        patched = model.clone()
        patched.set_model_post_input_patch(patch)
        patched.set_model_attn1_patch(ReferenceLLLiteAttentionPatch(
            patch, {"q": "self_attn_q_proj", "k": "self_attn_k_proj", "v": "self_attn_v_proj"}))
        patched.set_model_attn2_patch(ReferenceLLLiteAttentionPatch(patch, {"q": "cross_attn_q_proj"}))
        return io.NodeOutput(patched)


class AnimaMeshRegions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ExperimentalAnimaMeshRegions",
            display_name="Anima 3D Mesh → Regional Map",
            category="Anima Ref Adapter/experimental",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("palette", default="#ff4040,#4080ff,#40ff80,#ffdf40"),
                io.Float.Input("black_threshold", default=0.02, min=0.0, max=1.0, step=0.005),
            ],
            outputs=[io.Image.Output("regions")],
        )

    @classmethod
    def execute(cls, image, palette, black_threshold):
        colors = []
        for value in palette.split(","):
            value = value.strip().removeprefix("#")
            if len(value) != 6:
                raise ValueError("Use comma-separated six-digit RGB colors, e.g. #ff4040,#4080ff.")
            colors.append([int(value[index:index + 2], 16) / 255.0 for index in (0, 2, 4)])
        colors = torch.tensor(colors, device=image.device, dtype=image.dtype)
        rgb = image[..., :3]
        similarity = F.normalize(rgb, dim=-1) @ F.normalize(colors, dim=-1).T
        labels = similarity.argmax(dim=-1)
        regions = colors[labels]
        regions = regions * (rgb.amax(dim=-1, keepdim=True) > black_threshold)
        return io.NodeOutput(regions)


class AnimaLLLiteRefExtension(ComfyExtension):
    async def get_node_list(self):
        return [AnimaRefAdapterControlsLoader, AnimaLLLiteRefApply, AnimaMeshRegions]


async def comfy_entrypoint():
    return AnimaLLLiteRefExtension()
