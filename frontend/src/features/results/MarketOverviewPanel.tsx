/**
 * @file src/features/results/MarketOverviewPanel.tsx
 * @description Market overview section combining KPI cards and carrier bar charts.
 */

import { MetricCard } from "../../components/ui/MetricCard";
import { SimpleBarChart } from "../../components/charts/SimpleBarChart";
import { useCarrierLookup } from "../../hooks/useCarrierLookup";
import { getCarrierDisplayName } from "../../utils/carrierDisplay";
import {
  computeMarketOverview,
  getCarrierFareBars,
  getCarrierShareBars,
} from "./analytics";
import type { RouteFilters, RouteMarketPowerRow } from "../../types/data";
import { formatCurrency, formatNumber } from "../../utils/format";

interface MarketOverviewPanelProps {
  rows: RouteMarketPowerRow[];
  filters: RouteFilters;
}

export function MarketOverviewPanel({ rows }: MarketOverviewPanelProps) {
  // Lookup converts carrier codes into user-friendly names in chart labels.
  const carrierLookup = useCarrierLookup();
  // Compute KPI totals and chart rows from current filtered route rows.
  const stats = computeMarketOverview(rows);
  const shareBars = getCarrierShareBars(rows).map((item) => ({
    ...item,
    label: getCarrierDisplayName(item.label, carrierLookup),
  }));
  const fareBars = getCarrierFareBars(rows).map((item) => ({
    ...item,
    label: getCarrierDisplayName(item.label, carrierLookup),
  }));
  const formatCarrierShare = (value: number) => (value > 0 && value < 1 ? "< 1%" : `${formatNumber(value)} %`);

  return (
    <section className="panel-grid">
      <div className="metrics-grid">
        <MetricCard label="Total Passengers" value={formatNumber(stats.totalPassengers)} />
        <MetricCard label="Passenger-Weighted Avg Fare" value={formatCurrency(stats.avgFare)} />
        <MetricCard label="Number of Carriers" value={formatNumber(stats.carriers)} />
        <MetricCard label="Avg Route HHI" value={formatNumber(stats.avgHhi)} />
      </div>

      <div className="two-col">
        <SimpleBarChart title="Market Share by Carrier (%)" rows={shareBars} valueFormatter={formatCarrierShare} />
        <SimpleBarChart title="Average Fare by Carrier" rows={fareBars} color="var(--chart-2)" />
      </div>
    </section>
  );
}





