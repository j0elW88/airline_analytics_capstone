/**
 * @file src/components/ui/InteractiveTitle.tsx
 * @description Animated title renderer used for the branded landing heading.
 */

import type { CSSProperties } from "react";
import { useState } from "react";

interface InteractiveTitleProps {
  text: string;
  className?: string;
}

export function InteractiveTitle({ text, className = "" }: InteractiveTitleProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const words = text.trim().split(/\s+/);
  let runningIndex = 0;

  return (
    <h1
      onMouseLeave={() => setHoveredIndex(null)}
      className={`interactive-title ${className}`.trim()}
    >
      {words.map((word, wordIdx) => {
        const baseIndex = runningIndex;
        runningIndex += word.length + 1;

        return (
          <span key={`${word}-${wordIdx}`} className="interactive-title__word">
            {word.split("").map((char, charIdx) => {
              const idx = baseIndex + charIdx;
              const distance = hoveredIndex === null ? null : Math.abs(hoveredIndex - idx);
              let toneClass = "";

              if (distance === 0) {
                toneClass = "interactive-title__char--focus";
              } else if (distance === 1) {
                toneClass = "interactive-title__char--near";
              } else if (distance === 2) {
                toneClass = "interactive-title__char--mid";
              }

              return (
                <span
                  key={`${word}-${char}-${charIdx}`}
                  onMouseEnter={() => setHoveredIndex(idx)}
                  className={`interactive-title__char ${toneClass}`.trim()}
                  style={{ "--char-index": idx } as CSSProperties}
                >
                  {char}
                </span>
              );
            })}
          </span>
        );
      })}
    </h1>
  );
}





