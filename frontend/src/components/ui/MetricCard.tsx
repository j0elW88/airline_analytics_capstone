/**
 * @file src/components/ui/MetricCard.tsx
 * @description Compact KPI display card for primary analytics metrics.
 */

interface MetricCardProps {
  label: string;
  value: string;
  hint?: string;
  tooltip?: string;
}

export function MetricCard({ label, value, hint, tooltip }: MetricCardProps) {
  return (
    <article className="metric-card" title={tooltip}>
      <p className="metric-card__label">{label}</p>
      <p className="metric-card__value">{value}</p>
      {hint ? <p className="metric-card__hint">{hint}</p> : null}
    </article>
  );
}





