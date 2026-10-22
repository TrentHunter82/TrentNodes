import torch
import torch.nn.functional as F
import numpy as np


class AnimationDuplicateFrameProcessor:
    """
    A ComfyUI node that processes animation frames to replace duplicate
    sequences with gray frames, making animation timing structure visible.

    Enhanced with multiple similarity metrics, frame padding insertion,
    and frame tracking.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Animation frames to analyze for held (duplicate) frames."}),
                "similarity_method": (
                    ["hybrid", "ssim", "histogram", "perceptual"],
                    {
                        "default": "hybrid",
                        "tooltip": "Method for calculating frame similarity"
                    }
                ),
                "similarity_threshold": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "How similar frames need to be to count as duplicates"
                }),
                "motion_tolerance": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 0.3,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Tolerance for small movements/changes"
                }),
                "gray_style": (
                    ["solid_gray", "desaturated", "dimmed"],
                    {
                        "default": "desaturated",
                        "tooltip": "How to render the gray replacement frames"
                    }
                ),
                "gray_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Intensity of gray effect"
                }),
                "preserve_first": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep the first frame of each duplicate sequence "
                              "unchanged"
                }),
                "preserve_last": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep the last frame of each duplicate sequence "
                              "unchanged"
                }),
                "preserve_global_first": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep the very first frame of the entire batch "
                              "unchanged"
                }),
                "preserve_global_last": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep the very last frame of the entire batch "
                              "unchanged"
                }),
                "skip_second": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip the 2nd preserved frame from the list of "
                              "all preserved frames"
                }),
                "skip_second_to_last": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip the 2nd-to-last preserved frame from the "
                              "list of all preserved frames"
                }),
                "min_sequence_length": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Minimum frames in a sequence to consider as "
                              "duplicates"
                }),
                "min_gray_frames": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Minimum gray frames needed between preserved "
                              "frames (will insert extra if needed)"
                }),
                "insert_padding": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically insert extra gray frames when "
                              "sequences are too short"
                }),
                "align_keyframes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Align keyframes to multiples of alignment_multiple "
                              "for better video generation quality"
                }),
                "alignment_multiple": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Keyframes will be aligned to multiples of this "
                              "value (e.g., 4 means frames 0, 4, 8, 12...)"
                }),
            },
            "optional": {
                "debug_info": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print detailed analysis of duplicate sequences "
                              "found"
                }),
                "pad_to_4n_plus_1": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Append gray frames (mask white) until the batch "
                              "length is 4n+1, as WAN requires. The appended "
                              "frames are listed in removal_indices so the "
                              "Animation Frame Remover strips them again."
                }),
                "frame_multiple": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "The 4 in 4n+1. WAN and Hunyuan use 4, LTX uses 8."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = (
        "processed_frames", "duplicate_mask", "timing_report", "removal_indices"
    )
    FUNCTION = "process_animation_timing"
    CATEGORY = "Trent/Animation"
    DESCRIPTION = "Finds runs of duplicate (held) frames in an animation and replaces the repeats with gray frames plus a mask, so the timing is visible and the gaps can be regenerated. Can also insert padding frames and align keyframes to 4n+1-style grids."

    def calculate_ssim(self, frame1, frame2):
        """Calculate Structural Similarity Index between two frames."""
        gray1 = torch.sum(
            frame1 * torch.tensor([0.299, 0.587, 0.114]), dim=-1
        )
        gray2 = torch.sum(
            frame2 * torch.tensor([0.299, 0.587, 0.114]), dim=-1
        )

        gray1 = gray1.unsqueeze(0).unsqueeze(0)
        gray2 = gray2.unsqueeze(0).unsqueeze(0)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.avg_pool2d(gray1, 3, 1, 1)
        mu2 = F.avg_pool2d(gray2, 3, 1, 1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(gray1 * gray1, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(gray2 * gray2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(gray1 * gray2, 3, 1, 1) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()

    def calculate_histogram_similarity(self, frame1, frame2):
        """Calculate histogram similarity between two frames."""
        f1_np = frame1.cpu().numpy()
        f2_np = frame2.cpu().numpy()

        similarities = []
        for c in range(3):
            hist1 = np.histogram(f1_np[:, :, c], bins=32, range=(0, 1))[0]
            hist2 = np.histogram(f2_np[:, :, c], bins=32, range=(0, 1))[0]

            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)

            correlation = np.corrcoef(hist1, hist2)[0, 1]
            similarities.append(
                correlation if not np.isnan(correlation) else 0.0
            )

        return np.mean(similarities)

    def calculate_perceptual_similarity(self, frame1, frame2):
        """Calculate perceptual similarity using edge detection."""
        gray1 = torch.sum(
            frame1 * torch.tensor([0.299, 0.587, 0.114]), dim=-1
        )
        gray2 = torch.sum(
            frame2 * torch.tensor([0.299, 0.587, 0.114]), dim=-1
        )

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        gray1 = gray1.unsqueeze(0).unsqueeze(0)
        gray2 = gray2.unsqueeze(0).unsqueeze(0)

        edges1_x = F.conv2d(gray1, sobel_x, padding=1)
        edges1_y = F.conv2d(gray1, sobel_y, padding=1)
        edges1 = torch.sqrt(edges1_x**2 + edges1_y**2)

        edges2_x = F.conv2d(gray2, sobel_x, padding=1)
        edges2_y = F.conv2d(gray2, sobel_y, padding=1)
        edges2 = torch.sqrt(edges2_x**2 + edges2_y**2)

        flat1 = edges1.flatten()
        flat2 = edges2.flatten()

        correlation = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0))
        return correlation.item()

    def calculate_frame_similarity(self, frame1, frame2, method, motion_tolerance):
        """Calculate similarity between two frames using the specified method."""
        if method == "ssim":
            similarity = self.calculate_ssim(frame1, frame2)
        elif method == "histogram":
            similarity = self.calculate_histogram_similarity(frame1, frame2)
        elif method == "perceptual":
            similarity = self.calculate_perceptual_similarity(frame1, frame2)
        elif method == "hybrid":
            ssim_score = self.calculate_ssim(frame1, frame2)
            hist_score = self.calculate_histogram_similarity(frame1, frame2)
            perc_score = self.calculate_perceptual_similarity(frame1, frame2)
            similarity = 0.5 * ssim_score + 0.3 * hist_score + 0.2 * perc_score

        if similarity > (1.0 - motion_tolerance):
            similarity = min(1.0, similarity + motion_tolerance * 0.5)

        return similarity

    @staticmethod
    def next_valid_frame_count(count, multiple):
        """Smallest c >= count with c = multiple*n + 1 and n >= 1.

        Matches Wan21FrameAdjusterNode: the minimum is multiple + 1 (5 for
        WAN), and a count that already fits is returned unchanged.
        """
        count = max(count, multiple + 1)
        remainder = (count - 1) % multiple
        return count if remainder == 0 else count + (multiple - remainder)

    def create_gray_frame(self, original_frame, gray_style, gray_intensity):
        """Create a gray version of the frame based on the selected style."""
        if gray_style == "solid_gray":
            gray_frame = torch.full_like(original_frame, gray_intensity)
        elif gray_style == "desaturated":
            grayscale = torch.sum(
                original_frame * torch.tensor([0.299, 0.587, 0.114]),
                dim=-1, keepdim=True
            )
            grayscale = grayscale.expand(-1, -1, 3)
            gray_frame = (grayscale * gray_intensity +
                         original_frame * (1 - gray_intensity))
        elif gray_style == "dimmed":
            gray_frame = original_frame * gray_intensity

        return gray_frame

    def analyze_duplicate_sequences(self, images, similarity_threshold,
                                    motion_tolerance, similarity_method,
                                    min_sequence_length, debug_info):
        """Analyze the batch to find duplicate frame sequences."""
        batch_size = images.shape[0]

        similarities = []
        for i in range(1, batch_size):
            sim = self.calculate_frame_similarity(
                images[i-1], images[i], similarity_method, motion_tolerance
            )
            similarities.append(sim)

        if debug_info:
            print(f"\n=== Enhanced Animation Timing Analysis ===")
            print(f"Method: {similarity_method}, Threshold: {similarity_threshold}")
            print(f"Motion tolerance: {motion_tolerance}")
            print(f"Min sequence length: {min_sequence_length}")

        sequences = []
        current_sequence_start = 0
        current_sequence_length = 1

        for i, similarity in enumerate(similarities):
            frame_idx = i + 1

            if similarity >= similarity_threshold:
                current_sequence_length += 1
                if debug_info:
                    print(f"Frame {frame_idx}: SIMILAR (score: {similarity:.3f})")
            else:
                if current_sequence_length >= min_sequence_length:
                    sequences.append(
                        (current_sequence_start, current_sequence_length)
                    )

                    if debug_info:
                        end = current_sequence_start + current_sequence_length - 1
                        print(f"Sequence {current_sequence_start}-{end}: "
                              f"{current_sequence_length} frames")

                current_sequence_start = frame_idx
                current_sequence_length = 1

                if debug_info:
                    print(f"Frame {frame_idx}: NEW SCENE (score: {similarity:.3f})")

        if current_sequence_length >= min_sequence_length:
            sequences.append((current_sequence_start, current_sequence_length))

        report = f"Enhanced Animation Timing Analysis:\n"
        report += f"Method: {similarity_method}\n"
        report += f"Total frames: {batch_size}\n"
        report += f"Duplicate sequences found: {len(sequences)}\n"
        report += f"Min sequence length: {min_sequence_length}\n"

        if sequences:
            report += f"\nSequence breakdown:\n"
            for i, (start, length) in enumerate(sequences):
                report += (f"  Sequence {i+1}: frames {start}-{start+length-1} "
                          f"({length} frames)\n")

        if debug_info and similarities:
            avg_sim = np.mean(similarities)
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)
            report += f"\nSimilarity statistics:\n"
            report += f"  Average: {avg_sim:.3f}\n"
            report += f"  Min: {min_sim:.3f}\n"
            report += f"  Max: {max_sim:.3f}\n"

        return sequences, report

    def insert_padding_frames(self, images, sequences, preserve_first,
                              preserve_last, min_gray_frames, gray_style,
                              gray_intensity, debug_info):
        """Insert additional gray frames into sequences that are too short."""
        if min_gray_frames == 0:
            return images, [], sequences

        sequences_needing_padding = []
        for seq_idx, (start, length) in enumerate(sequences):
            gray_count = length
            if preserve_first and length > 1:
                gray_count -= 1
            if preserve_last and length > 1:
                gray_count -= 1

            if gray_count < min_gray_frames:
                frames_to_add = min_gray_frames - gray_count
                sequences_needing_padding.append(
                    (seq_idx, start, length, frames_to_add)
                )

        if not sequences_needing_padding:
            return images, [], sequences

        if debug_info:
            print(f"\n=== Frame Insertion Analysis ===")
            print(f"Sequences needing padding: {len(sequences_needing_padding)}")

        new_frames = []
        inserted_indices = []
        old_to_new_mapping = {}

        insertion_map = {}
        for seq_idx, start, length, frames_to_add in sequences_needing_padding:
            insert_after = start
            if insert_after not in insertion_map:
                insertion_map[insert_after] = 0
            insertion_map[insert_after] += frames_to_add

            if debug_info:
                print(f"Sequence {seq_idx} at {start}: will insert "
                      f"{frames_to_add} frames after index {insert_after}")

        new_idx = 0
        for original_idx in range(images.shape[0]):
            old_to_new_mapping[original_idx] = new_idx
            new_frames.append(images[original_idx])
            new_idx += 1

            if original_idx in insertion_map:
                frames_to_add = insertion_map[original_idx]
                gray_frame = self.create_gray_frame(
                    images[original_idx], gray_style, gray_intensity
                )

                for i in range(frames_to_add):
                    new_frames.append(gray_frame)
                    inserted_indices.append(new_idx)
                    new_idx += 1

                    if debug_info:
                        print(f"Inserted gray frame at new index {new_idx - 1}")

        padded_images = torch.stack(new_frames, dim=0)

        # Padding is inserted right after a run's first frame, so each run
        # grows by exactly the frames inserted for it.  Do NOT stretch a run
        # to the next run's start: that absorbs (and grays) the unique
        # frames sitting between two holds.
        adjusted_sequences = []
        for start, length in sequences:
            new_start = old_to_new_mapping[start]
            new_length = length + insertion_map.get(start, 0)
            adjusted_sequences.append((new_start, new_length))

        if debug_info:
            print(f"Original batch size: {images.shape[0]}")
            print(f"New batch size: {padded_images.shape[0]}")
            print(f"Total frames inserted: {len(inserted_indices)}")
            print(f"Adjusted sequences: {adjusted_sequences}")

        return padded_images, inserted_indices, adjusted_sequences

    def align_keyframes_to_multiples(self, images, mask_tensor, preserved_frames,
                                     inserted_indices, alignment_multiple,
                                     gray_style, gray_intensity, debug_info,
                                     gray_source=None):
        """Align keyframes to multiples by inserting gray padding frames.

        Follows the same pattern as insert_padding_frames() for consistency.

        ``images`` may already be grayed out; pass the un-grayed batch as
        ``gray_source`` (same indexing) so the inserted frames are built from
        original pixels instead of being desaturated/dimmed twice.
        """
        if gray_source is None:
            gray_source = images
        if alignment_multiple <= 1:
            return images, mask_tensor, inserted_indices, preserved_frames

        keyframe_positions = sorted(preserved_frames)
        if not keyframe_positions:
            return images, mask_tensor, inserted_indices, preserved_frames

        if debug_info:
            print(f"\n=== Keyframe Alignment Analysis ===")
            print(f"Alignment multiple: {alignment_multiple}")
            print(f"Original keyframe positions: {keyframe_positions}")

        # Calculate how many frames to insert before each keyframe
        insertion_map = {}
        cumulative_offset = 0
        aligned_positions = []

        for orig_pos in keyframe_positions:
            current_pos = orig_pos + cumulative_offset

            # Calculate next aligned position >= current position
            # Use integer math to avoid floating point issues
            # For position 7 with multiple 4: (7 + 3) // 4 * 4 = 8
            target_pos = (
                (current_pos + alignment_multiple - 1) // alignment_multiple
            ) * alignment_multiple

            # Special case: if current_pos is already aligned, keep it there
            if current_pos % alignment_multiple == 0:
                target_pos = current_pos

            # Handle collisions with previous keyframes
            while target_pos in aligned_positions:
                target_pos += alignment_multiple

            frames_needed = target_pos - current_pos
            if frames_needed > 0 and orig_pos > 0:
                # Insert AFTER the frame before the keyframe
                insert_after_idx = orig_pos - 1
                if insert_after_idx not in insertion_map:
                    insertion_map[insert_after_idx] = 0
                insertion_map[insert_after_idx] += frames_needed
                cumulative_offset += frames_needed

                if debug_info:
                    print(f"Keyframe at orig_pos={orig_pos}: "
                          f"current_pos={current_pos}, target={target_pos}, "
                          f"inserting {frames_needed} frames after idx "
                          f"{insert_after_idx}")
            elif debug_info:
                print(f"Keyframe at orig_pos={orig_pos}: "
                      f"current_pos={current_pos}, target={target_pos}, "
                      f"no insertion needed")

            aligned_positions.append(target_pos)

        if not insertion_map:
            if debug_info:
                print("No alignment adjustments needed")
            return images, mask_tensor, inserted_indices, keyframe_positions

        # Build new frame list (same pattern as insert_padding_frames)
        # NOTE: Alignment frames are NOT added to inserted_indices because
        # they should NOT be removed - removing them would undo the alignment.
        # Only the original min_gray_frames padding should be removable.
        new_frames = []
        new_mask = []
        old_to_new = {}
        new_idx = 0

        for original_idx in range(images.shape[0]):
            old_to_new[original_idx] = new_idx
            new_frames.append(images[original_idx])
            new_mask.append(mask_tensor[original_idx])
            new_idx += 1

            if original_idx in insertion_map:
                frames_to_add = insertion_map[original_idx]
                gray_frame = self.create_gray_frame(
                    gray_source[original_idx], gray_style, gray_intensity
                )
                gray_mask = torch.ones_like(mask_tensor[original_idx])

                for _ in range(frames_to_add):
                    new_frames.append(gray_frame)
                    new_mask.append(gray_mask)
                    # Don't add to removal list - alignment frames must stay
                    new_idx += 1

                    if debug_info:
                        print(f"Inserted alignment frame at new index "
                              f"{new_idx - 1} (permanent, not removable)")

        # Update inserted_indices to reflect new positions after alignment
        updated_inserted = [old_to_new[idx] for idx in inserted_indices
                           if idx in old_to_new]

        aligned_images = torch.stack(new_frames, dim=0)
        aligned_mask = torch.stack(new_mask, dim=0)

        # Calculate ACTUAL new positions of original keyframes using old_to_new
        # This is more accurate than aligned_positions which is theoretical
        actual_keyframe_positions = []
        for orig_pos in keyframe_positions:
            if orig_pos in old_to_new:
                actual_keyframe_positions.append(old_to_new[orig_pos])

        if debug_info:
            print(f"Original batch size: {images.shape[0]}")
            print(f"Aligned batch size: {aligned_images.shape[0]}")
            print(f"Target aligned positions: {aligned_positions}")
            print(f"Actual keyframe positions: {actual_keyframe_positions}")
            print(f"old_to_new mapping for keyframes: "
                  f"{[(k, old_to_new[k]) for k in keyframe_positions]}")

        # Return ACTUAL positions, not theoretical targets
        return aligned_images, aligned_mask, updated_inserted, actual_keyframe_positions

    def process_animation_timing(self, images, similarity_method,
                                 similarity_threshold, motion_tolerance,
                                 gray_style, gray_intensity, preserve_first,
                                 preserve_last, preserve_global_first,
                                 preserve_global_last, skip_second,
                                 skip_second_to_last, min_sequence_length,
                                 min_gray_frames, insert_padding,
                                 align_keyframes, alignment_multiple,
                                 debug_info=False, pad_to_4n_plus_1=False,
                                 frame_multiple=4):
        """Main processing function with enhanced duplicate detection.

        Pipeline:
          1. detect duplicate runs
          2. insert min_gray_frames padding inside runs that are too short
          3. classify every frame exactly once: keyframe / gray / untouched
          4. apply the skip_* options to the keyframe list
          5. gray out the batch and build the mask from that classification
          6. align keyframes last, so the index remap covers everything
          7. optionally append gray frames up to the next 4n+1 count

        Frames that lie outside every duplicate run are left untouched
        (mask black) but are not keyframes, matching the behaviour the node
        has always had with padding disabled.
        """
        if debug_info:
            print(f"Starting enhanced animation timing processing...")
            print(f"Input batch shape: {images.shape}")
            print(f"Insert padding: {insert_padding}, "
                  f"Min gray frames: {min_gray_frames}")

        sequences, report = self.analyze_duplicate_sequences(
            images, similarity_threshold, motion_tolerance, similarity_method,
            min_sequence_length, debug_info
        )

        inserted_indices = []
        if insert_padding and min_gray_frames > 0:
            images, inserted_indices, sequences = self.insert_padding_frames(
                images, sequences, preserve_first, preserve_last,
                min_gray_frames, gray_style, gray_intensity, debug_info
            )

            if inserted_indices:
                report += f"\nFrame Insertion:\n"
                report += f"Frames inserted for padding: {len(inserted_indices)}\n"
                report += f"New total frame count: {images.shape[0]}\n"

        batch_size, height, width, channels = images.shape
        inserted_set = set(inserted_indices)

        # ---- Step 3: classify every frame exactly once --------------------
        # keyframes -> kept as-is, mask black.  gray -> replaced, mask white.
        # Anything in neither set is outside every run: untouched, mask black.
        keyframes = set()
        gray = set(inserted_set)

        for start_frame, sequence_length in sequences:
            last_frame = start_frame + sequence_length - 1
            for frame_idx in range(start_frame, last_frame + 1):
                if frame_idx in inserted_set:
                    continue
                if ((preserve_first and frame_idx == start_frame) or
                        (preserve_last and frame_idx == last_frame)):
                    keyframes.add(frame_idx)
                else:
                    gray.add(frame_idx)

        # Global first/last are batch-level rules. They must apply whether or
        # not the frame happens to sit inside a duplicate run.  Padding is only
        # ever inserted *after* a frame, so index 0 is always an original
        # frame, but trailing padding can follow the last original frame.
        original_indices = [i for i in range(batch_size) if i not in inserted_set]
        global_first_idx = original_indices[0]
        global_last_idx = original_indices[-1]

        if preserve_global_first:
            keyframes.add(global_first_idx)
            gray.discard(global_first_idx)
        if preserve_global_last:
            keyframes.add(global_last_idx)
            gray.discard(global_last_idx)

        preserved_frames = sorted(keyframes)
        preserved_before_skip = len(preserved_frames)

        # ---- Step 4: skip options operate on the ordered keyframe list ----
        frames_to_skip = set()
        if skip_second and len(preserved_frames) >= 2:
            frames_to_skip.add(preserved_frames[1])
        if skip_second_to_last and len(preserved_frames) >= 2:
            frames_to_skip.add(preserved_frames[-2])

        for frame_idx in frames_to_skip:
            keyframes.discard(frame_idx)
            gray.add(frame_idx)
        preserved_frames = sorted(keyframes)

        if debug_info:
            print(f"\nPreserve first frame of sequences: {preserve_first}")
            print(f"Preserve last frame of sequences: {preserve_last}")
            print(f"Preserve global first frame (idx {global_first_idx}): "
                  f"{preserve_global_first}")
            print(f"Preserve global last frame (idx {global_last_idx}): "
                  f"{preserve_global_last}")
            print(f"Skip 2nd preserved frame: {skip_second}")
            print(f"Skip 2nd-to-last preserved frame: {skip_second_to_last}")
            print(f"Frames skipped from preserved list: {sorted(frames_to_skip)}")
            print(f"Keyframes before alignment: {preserved_frames}")

        # ---- Step 5: gray-out pass driven by the classification -----------
        processed_images = images.clone()
        mask_tensor = torch.zeros(
            batch_size, height, width,
            dtype=images.dtype, device=images.device
        )

        for frame_idx in sorted(gray):
            if frame_idx not in inserted_set:
                processed_images[frame_idx] = self.create_gray_frame(
                    images[frame_idx], gray_style, gray_intensity
                )
            mask_tensor[frame_idx] = 1.0

        if debug_info:
            print(f"\nProcessing {batch_size} frames "
                  f"({len(sequences)} duplicate sequences)...")
            for frame_idx in range(batch_size):
                if frame_idx in keyframes:
                    print(f"  Frame {frame_idx}: PRESERVED - mask BLACK")
                elif frame_idx in frames_to_skip:
                    print(f"  Frame {frame_idx}: SKIPPED (was preserved) "
                          "- mask WHITE")
                elif frame_idx in inserted_set:
                    print(f"  Frame {frame_idx}: GRAY (inserted padding) "
                          "- mask WHITE")
                elif frame_idx in gray:
                    print(f"  Frame {frame_idx}: GRAY (replaced) - mask WHITE")
                else:
                    print(f"  Frame {frame_idx}: UNTOUCHED (outside runs) "
                          "- mask BLACK")

        # ---- Step 6: keyframe alignment runs last ------------------------
        # It inserts permanent gray frames and remaps the keyframe list, the
        # removable padding indices and the mask in one pass, so nothing
        # downstream can hold stale indices.
        if align_keyframes and alignment_multiple > 1:
            (processed_images, mask_tensor, inserted_indices,
             preserved_frames) = self.align_keyframes_to_multiples(
                processed_images, mask_tensor, preserved_frames,
                inserted_indices, alignment_multiple, gray_style,
                gray_intensity, debug_info, gray_source=images
            )
            batch_size = processed_images.shape[0]

            # Always report alignment info when alignment is enabled
            report += f"\nKeyframe Alignment:\n"
            report += f"Alignment multiple: {alignment_multiple}\n"
            report += f"Aligned keyframe positions: {preserved_frames}\n"
            report += f"New total frame count after alignment: {batch_size}\n"

        # ---- Step 7: pad to multiple*n + 1 frames (WAN needs 4n+1) --------
        # Appended frames are gray, mask white, and go into removal_indices so
        # the Frame Remover restores the original count afterwards.
        appended_indices = []
        if pad_to_4n_plus_1:
            target_count = self.next_valid_frame_count(batch_size, frame_multiple)
            frames_to_add = target_count - batch_size
            if frames_to_add > 0:
                tail_frame = self.create_gray_frame(
                    images[global_last_idx], gray_style, gray_intensity
                )
                tail_frames = tail_frame.unsqueeze(0).repeat(
                    frames_to_add, 1, 1, 1
                )
                tail_mask = torch.ones(
                    frames_to_add, height, width,
                    dtype=mask_tensor.dtype, device=mask_tensor.device
                )
                processed_images = torch.cat([processed_images, tail_frames], dim=0)
                mask_tensor = torch.cat([mask_tensor, tail_mask], dim=0)
                appended_indices = list(range(batch_size, target_count))
                inserted_indices = sorted(inserted_indices) + appended_indices
                batch_size = target_count

            if debug_info:
                print(f"\n=== {frame_multiple}n+1 Frame Padding ===")
                print(f"Appended {frames_to_add} gray frames -> "
                      f"{batch_size} frames total")

            report += f"\n{frame_multiple}n+1 Frame Padding:\n"
            report += f"Frames appended: {frames_to_add}\n"
            report += f"Final frame count: {batch_size}\n"

        frame_is_gray = mask_tensor.reshape(batch_size, -1).amax(dim=1) > 0.5
        frames_processed = int(frame_is_gray.sum().item())

        if debug_info:
            print(f"\nMask Summary:")
            print(f"Total gray frames (white in mask): {frames_processed}")
            print(f"Total preserved frames (black in mask): "
                  f"{batch_size - frames_processed}")
            print(f"Final keyframe positions: {preserved_frames}")

        removal_indices_str = ",".join(map(str, sorted(inserted_indices)))

        report += f"\nProcessing results:\n"
        report += f"Frames replaced with gray: {frames_processed}\n"
        report += f"Method used: {similarity_method}\n"
        report += f"Motion tolerance: {motion_tolerance}\n"
        report += f"Gray style: {gray_style} (intensity: {gray_intensity})\n"
        report += f"Preserve first frame of sequences: {preserve_first}\n"
        report += f"Preserve last frame of sequences: {preserve_last}\n"
        report += f"Preserve global first frame: {preserve_global_first}\n"
        report += f"Preserve global last frame: {preserve_global_last}\n"
        report += f"Total preserved frames before skip: {preserved_before_skip}\n"
        report += f"Skip 2nd preserved frame: {skip_second}\n"
        report += f"Skip 2nd-to-last preserved frame: {skip_second_to_last}\n"
        if frames_to_skip:
            report += f"Frames skipped from preserved list: "
            report += f"{sorted(frames_to_skip)}\n"
        report += f"Keyframe alignment enabled: {align_keyframes}\n"
        if align_keyframes:
            report += f"Alignment multiple: {alignment_multiple}\n"
        report += f"Final keyframe positions: {preserved_frames}\n"

        if inserted_indices:
            report += f"\nRemoval Information:\n"
            report += f"Frames to remove after processing: {removal_indices_str}\n"
            report += f"(These are the padded frames that should be removed)\n"
            if appended_indices:
                report += (f"Of these, {len(appended_indices)} are the "
                           f"{frame_multiple}n+1 tail frames\n")

        return (processed_images, mask_tensor, report, removal_indices_str)


class AnimationFrameRemover:
    """
    A ComfyUI node that removes specified frames from an image batch.
    Used to remove padding frames inserted by AnimationDuplicateFrameProcessor.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Frame batch to remove frames from."}),
                "removal_indices": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated list of frame indices to remove "
                              "(from timing processor)"
                }),
            },
            "optional": {
                "debug_info": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print information about removed frames"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "removal_report")
    FUNCTION = "remove_frames"
    CATEGORY = "Trent/Animation"
    DESCRIPTION = "Removes frames at the given comma-separated indices from an image batch. Use it with the removal_indices output of Enhanced Animation Timing Processor to strip inserted padding frames."

    def remove_frames(self, images, removal_indices, debug_info=False):
        """Remove frames at the specified indices from the batch."""
        if not removal_indices or removal_indices.strip() == "":
            report = "No frames to remove"
            if debug_info:
                print(report)
            return (images, report)

        try:
            indices_to_remove = [
                int(idx.strip())
                for idx in removal_indices.split(",")
                if idx.strip()
            ]
            indices_to_remove = sorted(set(indices_to_remove))
        except ValueError as e:
            error_msg = f"Error parsing removal indices: {e}"
            print(error_msg)
            return (images, error_msg)

        if debug_info:
            print(f"Removing {len(indices_to_remove)} frames: {indices_to_remove}")

        batch_size = images.shape[0]
        valid_indices = [
            idx for idx in indices_to_remove if 0 <= idx < batch_size
        ]
        invalid_indices = [
            idx for idx in indices_to_remove if idx < 0 or idx >= batch_size
        ]

        if invalid_indices and debug_info:
            print(f"Warning: Invalid indices (out of range): {invalid_indices}")

        if not valid_indices:
            report = "No valid frames to remove"
            if debug_info:
                print(report)
            return (images, report)

        keep_mask = torch.ones(batch_size, dtype=torch.bool)
        for idx in valid_indices:
            keep_mask[idx] = False

        filtered_images = images[keep_mask]

        report = f"Frame Removal Report:\n"
        report += f"Original frame count: {batch_size}\n"
        report += f"Frames removed: {len(valid_indices)}\n"
        report += f"New frame count: {filtered_images.shape[0]}\n"
        report += f"Removed indices: {valid_indices}\n"

        if invalid_indices:
            report += f"Invalid indices (ignored): {invalid_indices}\n"

        if debug_info:
            print(report)

        return (filtered_images, report)


NODE_CLASS_MAPPINGS = {
    "AnimationDuplicateFrameProcessor": AnimationDuplicateFrameProcessor,
    "AnimationFrameRemover": AnimationFrameRemover
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimationDuplicateFrameProcessor": "Enhanced Animation Timing Processor",
    "AnimationFrameRemover": "Animation Frame Remover"
}
