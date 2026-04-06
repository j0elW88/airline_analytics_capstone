/**
 * @file src/pages/AnalyzeMultiPage.tsx
 * @description Screen for choosing multiple periods before opening analytics.
 */

import { useEffect, useMemo, useState } from "react";
import { PageShell } from "../components/layout/PageShell";
import { AppButton } from "../components/ui/AppButton";
import { EmptyState } from "../components/ui/EmptyState";

const MIN_PERIODS_REQUIRED = 2;
const MAX_PERIODS_ALLOWED = 5;

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
  const [selected, setSelected] = useState<string[]>(initialSelected.slice(0, MAX_PERIODS_ALLOWED));
  const selectedSet = useMemo(() => new Set(selected), [selected]);
  const canSelectMore = selected.length < MAX_PERIODS_ALLOWED;
  const hasMinimumPeriods = selected.length >= MIN_PERIODS_REQUIRED;

  useEffect(() => {
    setSelected((prev) => prev.filter((period) => periods.includes(period)).slice(0, MAX_PERIODS_ALLOWED));
  }, [periods]);

  function togglePeriod(period: string) {
    setSelected((prev) => {
      if (prev.includes(period)) {
        return prev.filter((item) => item !== period);
      }
      if (prev.length >= MAX_PERIODS_ALLOWED) {
        return prev;
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
          Compare mode prototype is integrated here. Select at least two periods to run the current
          multi-period comparative workflow.
        </p>
        <p className="muted analyze-multi-selection-status">
          Select {MIN_PERIODS_REQUIRED} to {MAX_PERIODS_ALLOWED} periods. Currently selected: {selected.length}.
        </p>
      </section>

      <section className="period-checklist">
        {periods.map((period) => (
          <label key={period} className="check-item">
            <input
              type="checkbox"
              checked={selectedSet.has(period)}
              disabled={!selectedSet.has(period) && !canSelectMore}
              onChange={() => togglePeriod(period)}
            />
            {period}
          </label>
        ))}
      </section>

      <div className="page-footer-actions">
        <AppButton variant="primary" onClick={() => onOpenAnalytics(selected)} disabled={!hasMinimumPeriods}>
          Open Multi-Period Analytics
        </AppButton>
        <AppButton onClick={onAddDataset}>Add Data Set</AppButton>
      </div>
    </PageShell>
  );
}





