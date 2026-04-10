import React from "react";
import { Composition } from "remotion";
import { AIFrameOverlay } from "./compositions/AIFrameOverlay";
import { TitleCard } from "./compositions/TitleCard";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="AIFrameOverlay"
        component={AIFrameOverlay}
        durationInFrames={81}
        fps={24}
        width={1280}
        height={720}
        defaultProps={{
          frameDir: "comfyui_frames",
          frameCount: 81,
          title: "Default Title",
          subtitle: "Default Subtitle",
          zeroPad: 4,
        }}
      />
      <Composition
        id="TitleCard"
        component={TitleCard}
        durationInFrames={72}
        fps={24}
        width={1280}
        height={720}
        defaultProps={{
          title: "My Video",
          subtitle: "",
          bgColor: "#0a0a0a",
          textColor: "#ffffff",
        }}
      />
    </>
  );
};
