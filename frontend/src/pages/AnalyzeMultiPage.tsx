/**
 * @file src/pages/AnalyzeMultiPage.tsx
 * @description Screen for choosing multiple periods before opening analytics.
 */

import { useEffect, useMemo, useState } from "react";
import { PageShell } from "../components/layout/PageShell";
import { AppButton } from "../components/ui/AppButton";
import { EmptyState } from "../components/ui/EmptyState";

interface AnalyzeMultiPageProps {
  periods: string[];
  initialSelected: string[];
  onOpenAnalytics: (periods: string[]) => void;
  onAddDataset: () => void;
}

export function AnalyzeMultiPage({
  periods,
  initialSelected,
  onOpenAnalytics,
  onAddDataset,
}: AnalyzeMultiPageProps) {
  const [selected, setSelected] = useState<string[]>(initialSelected);
  const selectedSet = useMemo(() => new Set(selected), [selected]);

  useEffect(() => {
    setSelected((prev) => prev.filter((period) => periods.includes(period)));
  }, [periods]);

  function togglePeriod(period: string) {
    setSelected((prev) => {
      if (prev.includes(period)) {
        return prev.filter((item) => item !== period);
      }
      return [...prev, period].sort();
    });
  }

  if (periods.length === 0) {
    return (
      <PageShell title="Analyze Multiple Periods" subtitle="Multi-period workflow">
        <EmptyState
          title="No complete periods available"
          description="Load at least one period first."
          action={<AppButton onClick={onAddDataset}>Add Data Set</AppButton>}
        />
      </PageShell>
    );
  }

  return (
    <PageShell title="Analyze Multiple Periods" subtitle="Select periods for compare-across-periods prototype flow">
      <section className="prototype-card">
        <p>
          Compare mode prototype is integrated here. Select one or more periods to run the current
          multi-period comparative workflow.
        </p>
      </section>

      <section className="period-checklist">
        {periods.map((period) => (
          <label key={period} className="check-item">
            <input
              type="checkbox"
              checked={selectedSet.has(period)}
              onChange={() => togglePeriod(period)}
            />
            {period}
          </label>
        ))}
      </section>

      <div className="page-footer-actions">
        <AppButton variant="primary" onClick={() => onOpenAnalytics(selected)} disabled={selected.length === 0}>
          Open Multi-Period Analytics
        </AppButton>
        <AppButton onClick={onAddDataset}>Add Data Set</AppButton>
      </div>
    </PageShell>
  );
}





