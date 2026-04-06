/**
 * @file src/components/ui/Tabs.tsx
 * @description Reusable tab switcher for analytics mode/view selection.
 */

interface TabOption {
  key: string;
  label: string;
}

interface TabsProps {
  options: TabOption[];
  activeKey: string;
  onChange: (key: string) => void;
  className?: string;
}

export function Tabs({ options, activeKey, onChange, className = "" }: TabsProps) {
  return (
    <div className={`tabs ${className}`.trim()} role="tablist" aria-label="Analytics tabs">
      {options.map((option) => {
        const active = option.key === activeKey;
        return (
          <button
            key={option.key}
            type="button"
            className={`tab ${active ? "tab--active" : ""}`}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(option.key)}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}





