import React from "react";
import {
  AbsoluteFill,
  Img,
  staticFile,
  useCurrentFrame,
  interpolate,
  Easing,
} from "remotion";

interface AIFrameOverlayProps {
  frameDir: string;
  frameCount: number;
  title: string;
  subtitle: string;
  zeroPad?: number;
}

export const AIFrameOverlay: React.FC<AIFrameOverlayProps> = ({
  frameDir,
  frameCount,
  title,
  subtitle,
  zeroPad = 4,
}) => {
  const frame = useCurrentFrame();
  const frameNum = String(Math.min(frame, frameCount - 1)).padStart(
    zeroPad,
    "0"
  );
  const src = staticFile(`${frameDir}/frame_${frameNum}.png`);

  const barY = interpolate(frame, [10, 30], [100, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });

  const textOpacity = interpolate(frame, [20, 40], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <Img
        src={src}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />

      <div
        style={{
          position: "absolute",
          bottom: 60,
          left: 0,
          right: 0,
          transform: `translateY(${barY}px)`,
        }}
      >
        <div
          style={{
            background:
              "linear-gradient(90deg, rgba(0,0,0,0.8) 0%, " +
              "rgba(0,0,0,0.6) 70%, transparent 100%)",
            padding: "16px 40px",
            maxWidth: "60%",
          }}
        >
          <div
            style={{
              color: "white",
              fontSize: 36,
              fontWeight: 700,
              opacity: textOpacity,
            }}
          >
            {title}
          </div>
          <div
            style={{
              color: "#ccc",
              fontSize: 22,
              fontWeight: 400,
              opacity: textOpacity,
              marginTop: 4,
            }}
          >
            {subtitle}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
