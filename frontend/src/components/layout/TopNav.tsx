/**
 * @file src/components/layout/TopNav.tsx
 * @description Global top navigation bar with back button and page controls.
 */

import { AppButton } from "../ui/AppButton";

interface TopNavProps {
  showBack: boolean;
  onBack: () => void;
}

export function TopNav({ showBack, onBack }: TopNavProps) {
  return (
    <nav className="top-nav">
      <div />
      {showBack ? (
        <AppButton variant="ghost" onClick={onBack}>
          Back
        </AppButton>
      ) : null}
    </nav>
  );
}





