/**
 * @file src/pages/LoadedDatasetsPage.tsx
 * @description Screen listing currently loaded datasets and readiness state.
 */

import { PageShell } from "../components/layout/PageShell";
import { AppButton } from "../components/ui/AppButton";
import { EmptyState } from "../components/ui/EmptyState";

interface LoadedDatasetsPageProps {
  periods: string[];
  onAdd: () => void;
}

export function LoadedDatasetsPage({ periods, onAdd }: LoadedDatasetsPageProps) {
  return (
    <PageShell title="Loaded Data Sets" subtitle="Periods available in local frontend store">
      {periods.length === 0 ? (
        <EmptyState
          title="No loaded periods"
          description="Upload route and hub market power CSV files for a period."
          action={<AppButton onClick={onAdd}>Add Data Set</AppButton>}
        />
      ) : (
        <section className="loaded-list">
          {periods.map((period) => (
            <article key={period} className="loaded-item loaded-item--ready">
              <strong>{period}</strong>
              <span>Ready</span>
            </article>
          ))}
        </section>
      )}
      <div className="page-footer-actions">
        <AppButton variant="primary" onClick={onAdd}>
          Add Data Set
        </AppButton>
      </div>
    </PageShell>
  );
}





