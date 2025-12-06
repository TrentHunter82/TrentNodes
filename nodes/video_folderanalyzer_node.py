import os
import cv2
import json
from pathlib import Path
from datetime import datetime

class VideoFolderAnalyzer:
    """
    A ComfyUI node that analyzes all video files in a specified folder
    and generates a detailed report with size, frame rate, and file type information.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter folder path containing video files"
                }),
                "include_subfolders": ("BOOLEAN", {"default": False}),
                "output_format": (["text", "json", "markdown"], {"default": "markdown"}),
            },
            "optional": {
                "file_extensions": ("STRING", {
                    "default": "mp4,avi,mov,mkv,wmv,flv,webm,m4v,mpg,mpeg",
                    "multiline": False,
                    "placeholder": "Comma-separated video extensions"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "json_data")
    FUNCTION = "analyze_videos"
    CATEGORY = "video/analysis"
    OUTPUT_NODE = True

    def analyze_videos(self, folder_path, include_subfolders, output_format, file_extensions="mp4,avi,mov,mkv,wmv,flv,webm,m4v,mpg,mpeg"):
        """
        Main function to analyze all videos in the specified folder
        """
        if not folder_path or not os.path.exists(folder_path):
            return ("Error: Invalid folder path provided", "{}")
        
        # Parse file extensions
        extensions = [f".{ext.strip().lower()}" for ext in file_extensions.split(",")]
        
        # Collect all video files
        video_files = self.collect_video_files(folder_path, extensions, include_subfolders)
        
        if not video_files:
            return (f"No video files found in {folder_path}", "{}")
        
        # Analyze each video file
        video_data = []
        total_size = 0
        errors = []
        
        for video_path in video_files:
            try:
                info = self.analyze_single_video(video_path)
                video_data.append(info)
                total_size += info['size_bytes']
            except Exception as e:
                errors.append({
                    'file': str(video_path),
                    'error': str(e)
                })
        
        # Create report based on selected format
        if output_format == "json":
            report = self.generate_json_report(video_data, total_size, errors)
            json_data = report
        elif output_format == "markdown":
            report = self.generate_markdown_report(video_data, total_size, errors)
            json_data = json.dumps({
                "videos": video_data,
                "total_size_bytes": total_size,
                "total_size_readable": self.format_file_size(total_size),
                "errors": errors
            }, indent=2)
        else:  # text format
            report = self.generate_text_report(video_data, total_size, errors)
            json_data = json.dumps({
                "videos": video_data,
                "total_size_bytes": total_size,
                "total_size_readable": self.format_file_size(total_size),
                "errors": errors
            }, indent=2)
        
        return (report, json_data)
    
    def collect_video_files(self, folder_path, extensions, include_subfolders):
        """
        Collect all video files from the folder
        """
        video_files = []
        path = Path(folder_path)
        
        if include_subfolders:
            for ext in extensions:
                video_files.extend(path.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                video_files.extend(path.glob(f"*{ext}"))
        
        return sorted(video_files)
    
    def analyze_single_video(self, video_path):
        """
        Analyze a single video file using OpenCV
        """
        cap = cv2.VideoCapture(str(video_path))
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # If frame count is invalid, try to count manually (slower but more accurate)
            if frame_count <= 0:
                frame_count = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset position
            
            # Calculate duration
            duration_seconds = frame_count / fps if fps > 0 else 0
            duration_formatted = self.format_duration(duration_seconds)
            
            # Get file information
            file_stats = os.stat(video_path)
            file_size = file_stats.st_size
            
            # Get codec information
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            return {
                'filename': video_path.name,
                'path': str(video_path),
                'file_type': video_path.suffix.upper()[1:],
                'codec': codec if codec.isprintable() else 'Unknown',
                'resolution': f"{width}x{height}",
                'width': width,
                'height': height,
                'fps': round(fps, 2),
                'frame_count': frame_count,
                'duration_seconds': round(duration_seconds, 2),
                'duration_formatted': duration_formatted,
                'size_bytes': file_size,
                'size_readable': self.format_file_size(file_size),
                'bitrate_kbps': round((file_size * 8) / (duration_seconds * 1000), 2) if duration_seconds > 0 else 0
            }
        finally:
            cap.release()
    
    def format_file_size(self, size_bytes):
        """
        Format file size in human-readable format
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def format_duration(self, seconds):
        """
        Format duration in HH:MM:SS format
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def generate_markdown_report(self, video_data, total_size, errors):
        """
        Generate a Markdown formatted report
        """
        report = []
        report.append("# 📹 Video Folder Analysis Report")
        report.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Videos:** {len(video_data)}")
        report.append(f"**Total Size:** {self.format_file_size(total_size)}")
        
        if video_data:
            # Calculate total frames
            total_frames = sum(v['frame_count'] for v in video_data)
            report.append(f"**Total Frames:** {total_frames:,}")
            
            report.append("\n## 📊 Video Files Summary\n")
            report.append("| File | Type | Resolution | FPS | Frames | Duration | Size | Bitrate |")
            report.append("|------|------|------------|-----|--------|----------|------|---------|")
            
            for video in video_data:
                report.append(f"| {video['filename'][:25]} | {video['file_type']} | "
                            f"{video['resolution']} | {video['fps']} | "
                            f"{video['frame_count']:,} | "
                            f"{video['duration_formatted']} | {video['size_readable']} | "
                            f"{video['bitrate_kbps']} kbps |")
            
            # Statistics
            report.append("\n## 📈 Statistics\n")
            
            # Resolution distribution
            resolutions = {}
            for video in video_data:
                res = video['resolution']
                resolutions[res] = resolutions.get(res, 0) + 1
            
            report.append("### Resolution Distribution")
            for res, count in sorted(resolutions.items(), key=lambda x: x[1], reverse=True):
                report.append(f"- {res}: {count} video(s)")
            
            # File type distribution
            file_types = {}
            for video in video_data:
                ft = video['file_type']
                file_types[ft] = file_types.get(ft, 0) + 1
            
            report.append("\n### File Type Distribution")
            for ft, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
                report.append(f"- {ft}: {count} video(s)")
            
            # FPS statistics
            fps_values = [v['fps'] for v in video_data]
            report.append(f"\n### Frame Rate Statistics")
            report.append(f"- Average FPS: {sum(fps_values)/len(fps_values):.2f}")
            report.append(f"- Min FPS: {min(fps_values):.2f}")
            report.append(f"- Max FPS: {max(fps_values):.2f}")
            
            # Frame count statistics
            frame_counts = [v['frame_count'] for v in video_data]
            report.append(f"\n### Frame Count Statistics")
            report.append(f"- Total Frames: {sum(frame_counts):,}")
            report.append(f"- Average Frames per Video: {sum(frame_counts)//len(frame_counts):,}")
            report.append(f"- Min Frames: {min(frame_counts):,}")
            report.append(f"- Max Frames: {max(frame_counts):,}")
        
        if errors:
            report.append("\n## ⚠️ Errors\n")
            for error in errors:
                report.append(f"- **{error['file']}**: {error['error']}")
        
        return "\n".join(report)
    
    def generate_text_report(self, video_data, total_size, errors):
        """
        Generate a plain text formatted report
        """
        report = []
        report.append("=" * 80)
        report.append("VIDEO FOLDER ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Videos: {len(video_data)}")
        report.append(f"Total Size: {self.format_file_size(total_size)}")
        
        if video_data:
            total_frames = sum(v['frame_count'] for v in video_data)
            report.append(f"Total Frames: {total_frames:,}")
        
        report.append("-" * 80)
        
        if video_data:
            for i, video in enumerate(video_data, 1):
                report.append(f"\n[{i}] {video['filename']}")
                report.append(f"    Type: {video['file_type']} | Codec: {video['codec']}")
                report.append(f"    Resolution: {video['resolution']} | FPS: {video['fps']}")
                report.append(f"    Frames: {video['frame_count']:,} | Duration: {video['duration_formatted']}")
                report.append(f"    Size: {video['size_readable']} | Bitrate: {video['bitrate_kbps']} kbps")
        
        if errors:
            report.append("\n" + "=" * 80)
            report.append("ERRORS:")
            report.append("-" * 80)
            for error in errors:
                report.append(f"File: {error['file']}")
                report.append(f"Error: {error['error']}")
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def generate_json_report(self, video_data, total_size, errors):
        """
        Generate a JSON formatted report
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_videos": len(video_data),
            "total_size_bytes": total_size,
            "total_size_readable": self.format_file_size(total_size),
            "total_frames": sum(v['frame_count'] for v in video_data) if video_data else 0,
            "videos": video_data,
            "errors": errors,
            "statistics": {
                "resolutions": {},
                "file_types": {},
                "fps": {
                    "average": 0,
                    "min": 0,
                    "max": 0
                },
                "frames": {
                    "total": 0,
                    "average": 0,
                    "min": 0,
                    "max": 0
                }
            }
        }
        
        if video_data:
            # Calculate statistics
            for video in video_data:
                res = video['resolution']
                report_data['statistics']['resolutions'][res] = \
                    report_data['statistics']['resolutions'].get(res, 0) + 1
                
                ft = video['file_type']
                report_data['statistics']['file_types'][ft] = \
                    report_data['statistics']['file_types'].get(ft, 0) + 1
            
            fps_values = [v['fps'] for v in video_data]
            report_data['statistics']['fps'] = {
                "average": round(sum(fps_values)/len(fps_values), 2),
                "min": min(fps_values),
                "max": max(fps_values)
            }
            
            frame_counts = [v['frame_count'] for v in video_data]
            report_data['statistics']['frames'] = {
                "total": sum(frame_counts),
                "average": sum(frame_counts) // len(frame_counts),
                "min": min(frame_counts),
                "max": max(frame_counts)
            }
        
        return json.dumps(report_data, indent=2)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VideoFolderAnalyzer": VideoFolderAnalyzer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFolderAnalyzer": "Video Folder Analyzer"
}