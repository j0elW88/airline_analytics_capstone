/**
 * @file src/pages/AnalyzeOnePage.tsx
 * @description Screen for choosing one period before opening analytics.
 */

import { useEffect, useMemo, useState } from "react";
import { PageShell } from "../components/layout/PageShell";
import { AppButton } from "../components/ui/AppButton";

interface AnalyzeOnePageProps {
  periods: string[];
  initialPeriod: string | null;
  onOpenAnalytics: (period: string) => void;
  onAddDataset: () => void;
}

export function AnalyzeOnePage({
  periods,
  initialPeriod,
  onOpenAnalytics,
  onAddDataset,
}: AnalyzeOnePageProps) {
  const fallback = useMemo(() => periods[0] ?? "", [periods]);
  const [selected, setSelected] = useState(initialPeriod ?? fallback);
  const hasPeriods = periods.length > 0;

  useEffect(() => {
    if (periods.length === 0) {
      setSelected("");
      return;
    }
    if (!selected || !periods.includes(selected)) {
      setSelected(initialPeriod && periods.includes(initialPeriod) ? initialPeriod : periods[0]);
    }
  }, [periods, initialPeriod, selected]);

  return (
    <PageShell title="Analyze One Period" subtitle="Choose one loaded period">
      <section className="card selection-card analyze-one-card">
        <header className="card__header">
          <div className="card__header-main">
            <h3 className="card__title">Available Periods</h3>
            <p className="card__subtitle">Select one loaded period to open analytics.</p>
          </div>
          <div className="card__header-right">
            <span className="analyze-one-count">{periods.length} ready</span>
          </div>
        </header>
        <div className="card__body">
          {hasPeriods ? (
            <>
              <section className="form-grid analyze-one-form">
                <label>
                  Loaded periods
                  <select
                    value={selected}
                    onChange={(event) => setSelected(event.target.value)}
                    disabled={!hasPeriods}
                  >
                    {periods.map((period) => (
                      <option key={period} value={period}>
                        {period}
                      </option>
                    ))}
                  </select>
                </label>
              </section>

              <div className="page-footer-actions page-footer-actions--right analyze-one-actions">
                <AppButton onClick={onAddDataset}>Add Data Set</AppButton>
                <AppButton variant="primary" onClick={() => onOpenAnalytics(selected)} disabled={!selected}>
                  Open Analytics
                </AppButton>
              </div>
            </>
          ) : (
            <section className="empty-state analyze-one-empty">
              <h3>No complete periods available</h3>
              <p>Load at least one period with route and hub files.</p>
              <div className="empty-state__action">
                <AppButton onClick={onAddDataset}>Add Data Set</AppButton>
              </div>
            </section>
          )}
        </div>
      </section>
    </PageShell>
  );
}





