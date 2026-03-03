/**
 * @file src/features/results/RouteHubInsightsPanel.tsx
 * @description Detailed route/hub insight section with tables, bars, and histogram displays.
 */

import { useEffect, useMemo, useState } from "react";
import { Card } from "../../components/ui/Card";
import { AppButton } from "../../components/ui/AppButton";
import { DataTable, type DataColumn } from "../../components/ui/DataTable";
import { HistogramCard } from "../../components/charts/HistogramCard";
import { SimpleBarChart } from "../../components/charts/SimpleBarChart";
import { useCarrierLookup } from "../../hooks/useCarrierLookup";
import { getCarrierDisplayName } from "../../utils/carrierDisplay";
import {
  getCarrierRouteBreakdown,
  getFareValues,
  getHighCostRoutes,
  getHubPassengerBars,
  getRouteMarketSnapshot,
  getTopRoutes,
  summarizeByCarrier,
  summarizeHubMarkets,
} from "./analytics";
import type { HubMarketPowerRow, RouteFilters, RouteMarketPowerRow } from "../../types/data";
import { formatCurrency, formatNumber, formatPercent } from "../../utils/format";

interface RouteHubInsightsPanelProps {
  routeRows: RouteMarketPowerRow[];
  hubRows: HubMarketPowerRow[];
  routeFilters: RouteFilters;
  view: "routexairline" | "hubxairline";
}

interface RouteRow {
  route: string;
  passengers: number;
}

interface CostRow {
  route: string;
  avgFare: number;
}

interface RouteSnapshotRow {
  route: string;
  carrierCount: number;
  avgFare: number;
  avgSharePct: number;
  fareGapVsMin: number;
}

interface CarrierRouteRow {
  route: string;
  passengers: number;
  avgFare: number;
}

interface CarrierRow {
  carrier: string;
  passengers: number;
  avgFare: number;
  revenueProxy: number;
  totalMileage: number;
  usPassengerShare: number;
  estimatedFlights: number;
}

interface HubRow {
  hub: string;
  passengers: number;
  avgFare: number;
  destinationsServed: number;
}

const routeColumns: DataColumn<RouteRow>[] = [
  { key: "route", header: "Route", render: (row) => row.route },
  { key: "passengers", header: "Passengers", render: (row) => formatNumber(row.passengers) },
];

const costColumns: DataColumn<CostRow>[] = [
  { key: "route", header: "Route", render: (row) => row.route },
  { key: "avgFare", header: "Avg Fare", render: (row) => formatCurrency(row.avgFare) },
];

const routeSnapshotColumns: DataColumn<RouteSnapshotRow>[] = [
  { key: "route", header: "Route", render: (row) => row.route },
  { key: "carrierCount", header: "Carriers", render: (row) => formatNumber(row.carrierCount) },
  { key: "avgFare", header: "Avg Fare", render: (row) => formatCurrency(row.avgFare) },
  { key: "avgShare", header: "Avg Share", render: (row) => `${row.avgSharePct.toFixed(1)}%` },
  { key: "fareGap", header: "Fare Gap vs Min", render: (row) => formatCurrency(row.fareGapVsMin) },
];

const carrierRouteColumns: DataColumn<CarrierRouteRow>[] = [
  { key: "route", header: "Route", render: (row) => row.route },
  { key: "passengers", header: "Passengers", render: (row) => formatNumber(row.passengers) },
  { key: "avgFare", header: "Avg Fare", render: (row) => formatCurrency(row.avgFare) },
];

const carrierColumns: DataColumn<CarrierRow>[] = [
  { key: "carrier", header: "Carrier", render: (row) => row.carrier },
  { key: "passengers", header: "Passengers", render: (row) => formatNumber(row.passengers) },
  { key: "avgFare", header: "Avg Fare", render: (row) => formatCurrency(row.avgFare) },
  { key: "revenueProxy", header: "Revenue Proxy", render: (row) => formatCurrency(row.revenueProxy) },
  { key: "totalMileage", header: "Total Mileage", render: (row) => formatNumber(row.totalMileage) },
  { key: "share", header: "Passenger Share", render: (row) => formatPercent(row.usPassengerShare) },
  { key: "flights", header: "Estimated Flights", render: (row) => formatNumber(row.estimatedFlights) },
];

const hubColumns: DataColumn<HubRow>[] = [
  { key: "hub", header: "Hub", render: (row) => row.hub },
  { key: "passengers", header: "Actual Passengers", render: (row) => formatNumber(row.passengers) },
  { key: "avgFare", header: "Avg Fare", render: (row) => formatCurrency(row.avgFare) },
  { key: "destinations", header: "Destinations", render: (row) => formatNumber(row.destinationsServed) },
];

export function RouteHubInsightsPanel({ routeRows, hubRows, routeFilters, view }: RouteHubInsightsPanelProps) {
  // Carrier lookup makes carrier codes readable in tables/charts.
  const carrierLookup = useCarrierLookup();
  // "See All" toggles are local to this panel and reset on context changes.
  const [showAllHubBars, setShowAllHubBars] = useState(false);
  const [showAllHubRows, setShowAllHubRows] = useState(false);

  const {
    topRoutes,
    highCostRoutes,
    carriers,
    hubs,
    routeSnapshot,
    carrierRoutes,
    hubBars,
    fareValues,
  } = useMemo(() => {
    // Defensive guards keep rendering stable even if incoming props are partially malformed.
    const safeRouteRows = Array.isArray(routeRows) ? routeRows : [];
    const safeHubRows = Array.isArray(hubRows) ? hubRows : [];

    try {
      // Build all derived datasets once per input change for predictable performance.
      return {
        topRoutes: getTopRoutes(safeRouteRows),
        highCostRoutes: getHighCostRoutes(safeRouteRows),
        carriers: summarizeByCarrier(safeRouteRows)
          .slice(0, 12)
          .map((row) => ({ ...row, carrier: getCarrierDisplayName(row.carrier, carrierLookup) })),
        hubs: summarizeHubMarkets(safeHubRows, safeRouteRows),
        routeSnapshot: getRouteMarketSnapshot(safeRouteRows).slice(0, 12),
        carrierRoutes: getCarrierRouteBreakdown(safeRouteRows).slice(0, 12),
        hubBars: getHubPassengerBars(safeHubRows, safeRouteRows, Number.MAX_SAFE_INTEGER),
        fareValues: getFareValues(safeRouteRows),
      };
    } catch {
      return {
        topRoutes: [],
        highCostRoutes: [],
        carriers: [],
        hubs: [],
        routeSnapshot: [],
        carrierRoutes: [],
        hubBars: [],
        fareValues: [],
      };
    }
  }, [routeRows, hubRows, carrierLookup]);

  const carrierFocused = Boolean(routeFilters.carrier);
  const destinationFocused = !carrierFocused && Boolean(routeFilters.dest);
  const routeInsightTitle = carrierFocused
    ? "Route x Airline Insights (Carrier-focused display)"
    : destinationFocused
      ? "Route x Airline Insights (Destination-focused display)"
      : "Route x Airline Insights";

  useEffect(() => {
    // Reset expansion state whenever user changes view/filter context.
    setShowAllHubBars(false);
    setShowAllHubRows(false);
  }, [view, routeFilters.carrier, hubBars.length, hubs.length]);

  const displayedHubBars = showAllHubBars ? hubBars : hubBars.slice(0, 20);
  const displayedHubRows = showAllHubRows ? hubs : hubs.slice(0, 20);

  if (view === "hubxairline") {
    // Hub tab: prioritize hub passenger ranking and full hub-by-airline table.
    return (
      <section className="panel-grid">
        <SimpleBarChart
          title="Top Hubs by Actual Passengers"
          subtitle="Actual passenger counts shown here reflect the DB1B 10% market sample."
          rows={displayedHubBars}
          color="var(--chart-3)"
          headerRight={
            hubBars.length > 20 ? (
              <AppButton variant="neutral" onClick={() => setShowAllHubBars((prev) => !prev)}>
                {showAllHubBars ? "Show Top 20" : "See All"}
              </AppButton>
            ) : null
          }
        />
        <Card
          title="Hub x Airline Market Display"
          headerRight={
            hubs.length > 20 ? (
              <AppButton variant="neutral" onClick={() => setShowAllHubRows((prev) => !prev)}>
                {showAllHubRows ? "Show Top 20" : "See All"}
              </AppButton>
            ) : null
          }
        >
          <DataTable rows={displayedHubRows} columns={hubColumns} rowKey={(row) => row.hub} />
        </Card>
      </section>
    );
  }

  return (
    // Route tab: route snapshots, fare distribution, rankings, and carrier summary.
    <section className="panel-grid">
      <Card title={routeInsightTitle}>
        {carrierFocused ? (
          <DataTable rows={carrierRoutes} columns={carrierRouteColumns} rowKey={(row) => row.route} />
        ) : (
          <DataTable rows={routeSnapshot} columns={routeSnapshotColumns} rowKey={(row) => row.route} />
        )}
      </Card>

      <HistogramCard title="Fare Distribution" values={fareValues} />

      <div className="two-col">
        <Card title="Route Demand Rankings">
          <DataTable rows={topRoutes} columns={routeColumns} rowKey={(row) => row.route} />
        </Card>
        <Card title="High-Cost Routes">
          <DataTable rows={highCostRoutes} columns={costColumns} rowKey={(row) => row.route} />
        </Card>
      </div>

      <Card title="Carrier Summary">
        <DataTable rows={carriers} columns={carrierColumns} rowKey={(row) => row.carrier} />
      </Card>
    </section>
  );
}





