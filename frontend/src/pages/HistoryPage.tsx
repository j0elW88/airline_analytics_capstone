/**
 * @file src/pages/HistoryPage.tsx
 * @description Screen showing recent user actions and analysis history.
 */

import { PageShell } from "../components/layout/PageShell";
import { EmptyState } from "../components/ui/EmptyState";

interface HistoryPageProps {
  items: string[];
}

export function HistoryPage({ items }: HistoryPageProps) {
  return (
    <PageShell title="Session History" subtitle="Recent frontend actions">
      {items.length === 0 ? (
        <EmptyState title="No history yet" description="Load or analyze a dataset to create history events." />
      ) : (
        <ol className="history-list">
          {items.map((item, index) => (
            <li key={`${item}-${index}`}>{item}</li>
          ))}
        </ol>
      )}
    </PageShell>
  );
}





