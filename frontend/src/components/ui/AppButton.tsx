/**
 * @file src/components/ui/AppButton.tsx
 * @description Shared button primitive with style variants used across all pages.
 */

import type { ButtonHTMLAttributes } from "react";

type ButtonVariant = "default" | "primary" | "danger" | "ghost";
type ExtendedButtonVariant = ButtonVariant | "neutral" | "reverse" | "noShadow";

interface AppButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ExtendedButtonVariant;
  block?: boolean;
}

export function AppButton({
  variant = "default",
  block = false,
  className = "",
  children,
  ...props
}: AppButtonProps) {
  return (
    <button
      className={`app-button app-button--${variant} ${block ? "app-button--block" : ""} ${className}`.trim()}
      {...props}
    >
      {children}
    </button>
  );
}





