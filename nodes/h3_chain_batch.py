"""Count H3 extensions and retain their audio across VHS requeues."""

import logging

import torch


class H3ChainStep:
    CATEGORY = "Trent/MiniMax"
    DISPLAY_NAME = "H3 Chain Step"
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("step", "seed", "load_slot", "save_slot", "primer")
    FUNCTION = "advance"
    DESCRIPTION = ("Run an exact number of extensions with a VHS Meta Batch Manager. "
                   "Replaces the driver video. Connect seed to RandomNoise and step "
                   "to H3 Chain Audio + Save Barrier. For fixed tail slots, the primer "
                   "saves clip_index 1; wire load_slot and save_slot to the Load and "
                   "Save Latent clip_index inputs. With primer_pass, pass 0 is the "
                   "primer and the primer output switches the graph to it.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "meta_batch": ("VHS_BatchManager", {"forceInput": True,
                    "tooltip": "VHS Meta Batch Manager that requeues the graph once per pass. Connect no other batch loaders to it."}),
                "chunks": ("INT", {"default": 3, "min": 1, "max": 1000,
                    "tooltip": "New extension clips, excluding the primer. The supplied 141/39-frame workflow adds 4.25 seconds per chunk."}),
                "seed": ("INT", {"default": 987654321, "min": 0,
                                  "max": 0xffffffffffffffff,
                    "tooltip": "Base seed. Each pass outputs seed + step, so every extension gets a different seed."}),
            },
            "optional": {
                "primer_pass": ("BOOLEAN", {"default": False,
                    "tooltip": "Generate the primer as pass 0 of this chain, before the chunks. Pass 0 saves slot 1; pass k loads slot k and saves slot k + 1."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def advance(self, meta_batch, chunks, seed, prompt, unique_id, primer_pass=False):
        passes = chunks + 1 if primer_pass else chunks
        uid = str(prompt[str(unique_id)]["inputs"]["meta_batch"][0])
        step = int(prompt[uid]["inputs"].get("requeue", 0))
        if not 0 <= step < passes:
            raise ValueError("H3 chain exceeded its chunk count; queue a fresh run.")
        if meta_batch.inputs:
            raise ValueError("H3 Chain Step replaces the driver video. Disconnect other batch loaders.")
        if step == 0:
            # VHS resets its ID after finalizing. Its cached output may survive
            # into another manually queued job with identical manager widgets.
            meta_batch.reset()
            meta_batch.unique_id = uid
        elif meta_batch.unique_id != uid:
            raise ValueError("H3 batch state was lost. Queue a fresh run.")
        meta_batch.total_frames = passes * meta_batch.frames_per_batch
        meta_batch.has_closed_inputs = step == passes - 1
        logging.info("[H3 Chain] pass %d/%d", step + 1, passes)
        load_slot = step if primer_pass else step + 1
        return (step, (seed + step) % (1 << 64), load_slot, load_slot + 1,
                primer_pass and step == 0)


class H3ChainAudio:
    CATEGORY = "Trent/MiniMax"
    DISPLAY_NAME = "H3 Chain Audio + Save Barrier"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "accumulate"
    OUTPUT_NODE = True
    DESCRIPTION = ("Keep every extension's audio for the final VHS mux. Connect "
                   "trimmed audio and Save Latent's path; send this output to the "
                   "same batch manager's VHS Video Combine.")

    def __init__(self):
        self.chunks = []
        self.manager = None
        self.sample_rate = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO", {"tooltip": "Trimmed audio of the current extension. Chunks join into one track on the final pass."}),
            "meta_batch": ("VHS_BatchManager", {"forceInput": True,
                "tooltip": "Same VHS Meta Batch Manager as H3 Chain Step"}),
            "step": ("INT", {"forceInput": True,
                "tooltip": "Connect the step output of H3 Chain Step"}),
            "saved_tail": ("STRING", {"forceInput": True,
                "tooltip": "Connect Save Latent's path so the tail is saved before the next requeue."}),
        }}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def accumulate(self, audio, meta_batch, step, saved_tail):
        if step == 0:
            self.chunks = []
            self.manager = meta_batch
            self.sample_rate = int(audio["sample_rate"])
        if self.manager is not meta_batch or step != len(self.chunks):
            raise ValueError("H3 chain audio lost a chunk. Queue a fresh run; do not resume a partial batch.")
        if not saved_tail:
            raise ValueError("H3 chain requires the saved tail path before requeueing.")
        wave = audio["waveform"]
        if int(audio["sample_rate"]) != self.sample_rate:
            raise ValueError("H3 chain sample rate changed between chunks.")
        if self.chunks and wave.shape[:-1] != self.chunks[0].shape[:-1]:
            raise ValueError("H3 chain audio channels changed between chunks.")
        # Own a CPU copy: a slice can otherwise retain the full decoded GPU audio.
        self.chunks.append(wave.detach().to(device="cpu", dtype=torch.float32, copy=True))
        if meta_batch.has_closed_inputs:
            result = {"waveform": torch.cat(self.chunks, dim=-1),
                      "sample_rate": self.sample_rate}
            self.chunks = []
            self.manager = None
        else:
            # VHS ignores audio until the final pass, when it muxes the whole track.
            result = {"waveform": self.chunks[-1], "sample_rate": self.sample_rate}
        logging.info("[H3 Chain] audio chunk %d committed after %s", step + 1, saved_tail)
        return (result,)


NODE_CLASS_MAPPINGS = {"H3ChainStep": H3ChainStep, "H3ChainAudio": H3ChainAudio}
NODE_DISPLAY_NAME_MAPPINGS = {k: v.DISPLAY_NAME for k, v in NODE_CLASS_MAPPINGS.items()}
