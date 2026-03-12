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
  const routeMarketCount = new Set(
    rows.map((row) => `${String(row.Origin ?? "").trim().toUpperCase()}-${String(row.Dest ?? "").trim().toUpperCase()}`),
  ).size;

  return (
    <section className="panel-grid">
      <div className="metrics-grid">
        <MetricCard
          label="Total Passengers"
          value={formatNumber(stats.totalPassengers)}
          tooltip={`Total Passengers\nSum of total_passengers across filtered route rows.\nContributing rows: ${formatNumber(rows.length)}\nRoute markets: ${formatNumber(routeMarketCount)}`}
        />
        <MetricCard
          label="Passenger-Weighted Avg Fare"
          value={formatCurrency(stats.avgFare)}
          tooltip={`Passenger-Weighted Avg Fare\nCalculated as sum(fare x passengers) / sum(passengers).\nContributing rows: ${formatNumber(rows.length)}\nRoute markets: ${formatNumber(routeMarketCount)}\nPassengers in denominator: ${formatNumber(stats.totalPassengers)}`}
        />
        <MetricCard
          label="Number of Carriers"
          value={formatNumber(stats.carriers)}
          tooltip={`Number of Carriers\nCount of unique carriers after filters.\nContributing rows: ${formatNumber(rows.length)}`}
        />
        <MetricCard
          label="Avg Route HHI"
          value={formatNumber(stats.avgHhi)}
          tooltip={`Avg Route HHI\nAverage HHI across unique filtered route markets.\nContributing route markets: ${formatNumber(routeMarketCount)}`}
        />
      </div>

      <div className="two-col">
        <SimpleBarChart title="Market Share by Carrier (%)" rows={shareBars} valueFormatter={formatCarrierShare} />
        <SimpleBarChart title="Average Fare by Carrier" rows={fareBars} color="var(--chart-2)" />
      </div>
    </section>
  );
}





