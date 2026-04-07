/**
 * @file src/components/charts/SimpleBarChart.tsx
 * @description Reusable horizontal bar chart component used throughout analytics views.
 */

import type { ReactNode } from "react";
import { Card } from "../ui/Card";
import { formatNumber } from "../../utils/format";

interface BarDatum {
  label: string;
  value: number;
}

interface SimpleBarChartProps {
  title: string;
  subtitle?: string;
  rows: BarDatum[];
  color?: string;
  maxValue?: number;
  valueLabel?: string;
  valueFormatter?: (value: number) => string;
  headerRight?: ReactNode;
}

export function SimpleBarChart({
  title,
  subtitle,
  rows,
  color = "var(--chart-1)",
  maxValue,
  valueLabel,
  valueFormatter,
  headerRight,
}: SimpleBarChartProps) {
  const providedMaxValue = typeof maxValue === "number" && Number.isFinite(maxValue) && maxValue > 0
    ? maxValue
    : null;
  const resolvedMaxValue = providedMaxValue
    ? providedMaxValue
    : Math.max(...rows.map((item) => item.value), 0);

  return (
    <Card title={title} subtitle={subtitle} className="chart-card" headerRight={headerRight}>
      {rows.length === 0 ? (
        <p className="muted">No data for current selection.</p>
      ) : (
        <div className="bar-chart">
          {rows.map((row) => {
            const safeValue = Number.isFinite(row.value) ? Math.max(row.value, 0) : 0;
            const ratio = resolvedMaxValue > 0 ? (safeValue / resolvedMaxValue) * 100 : 0;
            const width = ratio > 0 ? `${Math.max(Math.min(ratio, 100), 2)}%` : "0%";
            return (
              <div key={row.label} className="bar-chart__row">
                <div className="bar-chart__meta">
                  <span>{row.label}</span>
                  <span>
                    {valueFormatter
                      ? valueFormatter(row.value)
                      : `${formatNumber(row.value)}${valueLabel ? ` ${valueLabel}` : ""}`}
                  </span>
                </div>
                <div className="bar-chart__track">
                  <div className="bar-chart__fill" style={{ width, background: color }} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}





