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
import { formatRouteDisplay } from "../../utils/airports";
import {
  formatLocationSelectionLabel,
  parseLocationSelection,
} from "../../utils/locationTaxonomy";
import {
  fetchRouteFareDistribution,
  type FareDistributionBin,
  type RouteFareDistributionResponse,
} from "../../services/localBackend";
import {
  getCarrierRouteBreakdown,
  getFareDistributionPoints,
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
  period?: string;
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
  carrierTooltip: string;
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
  {
    key: "carrierCount",
    header: "Carriers",
    render: (row) => <span title={row.carrierTooltip}>{formatNumber(row.carrierCount)}</span>,
  },
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

const fareDistributionCarrierColors = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "#2f855a",
  "#b7791f",
  "#1f4e79",
  "#8b3d3d",
  "#5f3dc4",
];

interface ConsolidatedFareBin {
  fareStart: number;
  fareEnd: number;
  passengers: number;
  rowCount: number;
  sourceBinCount: number;
}

function consolidateFareBins(
  bins: FareDistributionBin[],
  totalPassengers: number,
  minFare: number,
  maxFare: number,
): ConsolidatedFareBin[] {
  const sorted = [...bins].sort((a, b) => a.fareStart - b.fareStart);
  if (sorted.length <= 1) {
    return sorted.map((bin) => ({
      fareStart: bin.fareStart,
      fareEnd: bin.fareEnd,
      passengers: bin.passengers,
      rowCount: bin.rowCount,
      sourceBinCount: 1,
    }));
  }

  const variation = Math.max(0, maxFare - minFare);
  let targetBinCount = 10;
  if (totalPassengers >= 12_000) {
    targetBinCount = 24;
  } else if (totalPassengers >= 6_000) {
    targetBinCount = 20;
  } else if (totalPassengers >= 3_000) {
    targetBinCount = 16;
  } else if (totalPassengers >= 1_500) {
    targetBinCount = 14;
  }
  if (variation >= 220) {
    targetBinCount += 4;
  }
  if (variation >= 320) {
    targetBinCount += 4;
  }
  targetBinCount = Math.max(8, Math.min(32, targetBinCount));

  if (sorted.length <= targetBinCount) {
    return sorted.map((bin) => ({
      fareStart: bin.fareStart,
      fareEnd: bin.fareEnd,
      passengers: bin.passengers,
      rowCount: bin.rowCount,
      sourceBinCount: 1,
    }));
  }

  const groupSize = Math.ceil(sorted.length / targetBinCount);
  const consolidated: ConsolidatedFareBin[] = [];
  for (let i = 0; i < sorted.length; i += groupSize) {
    const chunk = sorted.slice(i, i + groupSize);
    consolidated.push({
      fareStart: chunk[0].fareStart,
      fareEnd: chunk[chunk.length - 1].fareEnd,
      passengers: chunk.reduce((sum, bin) => sum + bin.passengers, 0),
      rowCount: chunk.reduce((sum, bin) => sum + bin.rowCount, 0),
      sourceBinCount: chunk.length,
    });
  }
  return consolidated;
}

interface FareDistributionPanelState {
  status: "idle" | "loading" | "ready" | "error";
  key: string;
  data: RouteFareDistributionResponse | null;
  error: string;
}

export function RouteHubInsightsPanel({ period, routeRows, hubRows, routeFilters, view }: RouteHubInsightsPanelProps) {
  // Carrier lookup makes carrier codes readable in tables/charts.
  const carrierLookup = useCarrierLookup();
  // "See All" toggles are local to this panel and reset on context changes.
  const [showAllHubBars, setShowAllHubBars] = useState(false);
  const [showAllHubRows, setShowAllHubRows] = useState(false);
  const [fareDistributionPanel, setFareDistributionPanel] = useState<FareDistributionPanelState>({
    status: "idle",
    key: "",
    data: null,
    error: "",
  });

  const originSelection = parseLocationSelection(routeFilters.origin);
  const destSelection = parseLocationSelection(routeFilters.dest);
  const routeSpecificScope = (
    view === "routexairline"
    && Boolean(period)
    && originSelection.type === "airport"
    && destSelection.type === "airport"
  );
  const useRouteContributorsInFareTooltip = (
    view === "routexairline"
    && !(originSelection.type === "airport" && destSelection.type === "airport")
  );
  const fareDistributionScopeKey = `${period ?? ""}|${routeFilters.origin}|${routeFilters.dest}|${routeFilters.carrier}`;

  const {
    topRoutes,
    highCostRoutes,
    carriers,
    hubs,
    routeSnapshot,
    carrierRoutes,
    hubBars,
    farePoints,
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
        routeSnapshot: getRouteMarketSnapshot(safeRouteRows)
          .slice(0, 12)
          .map((row) => ({
            ...row,
            carrierTooltip: row.carriers
              .map((carrierCode) => getCarrierDisplayName(carrierCode, carrierLookup))
              .join(", "),
          })),
        carrierRoutes: getCarrierRouteBreakdown(safeRouteRows).slice(0, 12),
        hubBars: getHubPassengerBars(safeHubRows, safeRouteRows, Number.MAX_SAFE_INTEGER),
        farePoints: getFareDistributionPoints(safeRouteRows).map((point) => ({
          value: point.value,
          label: useRouteContributorsInFareTooltip
            ? `${point.origin} -> ${point.dest}`
            : getCarrierDisplayName(point.carrier, carrierLookup),
          weight: point.weight,
        })),
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
        farePoints: [],
      };
    }
  }, [routeRows, hubRows, carrierLookup, useRouteContributorsInFareTooltip]);

  const carrierFocused = Boolean(routeFilters.carrier);
  const destinationFocused = !carrierFocused && destSelection.type !== "all";
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

  useEffect(() => {
    if (!routeSpecificScope) {
      setFareDistributionPanel((prev) => ({
        ...prev,
        status: "idle",
        key: "",
        data: null,
        error: "",
      }));
      return;
    }
    if (fareDistributionPanel.key !== fareDistributionScopeKey) {
      setFareDistributionPanel({
        status: "idle",
        key: fareDistributionScopeKey,
        data: null,
        error: "",
      });
    }
  }, [routeSpecificScope, fareDistributionScopeKey, fareDistributionPanel.key]);

  const displayedHubBars = showAllHubBars ? hubBars : hubBars.slice(0, 20);
  const displayedHubRows = showAllHubRows ? hubs : hubs.slice(0, 20);
  const standardFareBucketCount = useMemo(() => {
    if (farePoints.length === 0) {
      return 8;
    }
    let minFare = Number.POSITIVE_INFINITY;
    let maxFare = Number.NEGATIVE_INFINITY;
    farePoints.forEach((point) => {
      if (!Number.isFinite(point.value) || point.value <= 0) {
        return;
      }
      minFare = Math.min(minFare, point.value);
      maxFare = Math.max(maxFare, point.value);
    });
    const range = Number.isFinite(minFare) && Number.isFinite(maxFare)
      ? Math.max(0, maxFare - minFare)
      : 0;
    if (range >= 700) {
      return 16;
    }
    if (range >= 500) {
      return 14;
    }
    if (range >= 320) {
      return 12;
    }
    if (range >= 180) {
      return 10;
    }
    return 8;
  }, [farePoints]);

  async function handleLoadRouteFareDistribution() {
    if (!routeSpecificScope || !period) {
      return;
    }
    setFareDistributionPanel({
      status: "loading",
      key: fareDistributionScopeKey,
      data: null,
      error: "",
    });
    try {
      const payload = await fetchRouteFareDistribution({
        period,
        origin: originSelection.code,
        dest: destSelection.code,
        carrier: routeFilters.carrier || undefined,
      });
      setFareDistributionPanel({
        status: "ready",
        key: fareDistributionScopeKey,
        data: payload,
        error: "",
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed loading route fare distribution.";
      setFareDistributionPanel({
        status: "error",
        key: fareDistributionScopeKey,
        data: null,
        error: message,
      });
    }
  }

  function renderRouteSpecificFareDistribution(): JSX.Element {
    const titleBase = `Route Fare Variation: ${formatRouteDisplay(originSelection.code, destSelection.code)}`;
    if (fareDistributionPanel.status === "loading") {
      return (
        <Card title={titleBase}>
          <p className="muted">Computing route fare variation from cached DB1B distribution data...</p>
        </Card>
      );
    }
    if (fareDistributionPanel.status === "error") {
      return (
        <Card title={titleBase}>
          <p className="muted">{fareDistributionPanel.error}</p>
          <div className="load-import-actions">
            <AppButton variant="primary" onClick={handleLoadRouteFareDistribution}>Retry Load</AppButton>
          </div>
        </Card>
      );
    }
    if (fareDistributionPanel.status !== "ready" || !fareDistributionPanel.data) {
      return (
        <Card title={titleBase}>
          <p className="muted">
            This analysis is available only when a specific Origin airport and Destination airport are selected.
            Click below to load route-specific fare variation charts.
          </p>
          <div className="load-import-actions">
            <AppButton variant="primary" onClick={handleLoadRouteFareDistribution}>Load Fare Variation</AppButton>
          </div>
        </Card>
      );
    }

    const carriers = fareDistributionPanel.data.carriers;
    if (carriers.length === 0) {
      return (
        <Card title={titleBase}>
          <p className="muted">No fare variation data is available for this route scope.</p>
        </Card>
      );
    }

    const carrierCharts = carriers.map((carrier, index) => {
      const color = fareDistributionCarrierColors[index % fareDistributionCarrierColors.length];
      const consolidatedBins = consolidateFareBins(
        carrier.bins,
        carrier.totalPassengers,
        carrier.minFare,
        carrier.maxFare,
      );
      return {
        carrier,
        color,
        consolidatedBins,
      };
    });

    return (
      <section className="panel-grid">
        {carrierCharts.map(({ carrier, color, consolidatedBins }) => {
          const subtitle = `Passengers ${formatNumber(carrier.totalPassengers)} | Entries ${formatNumber(carrier.totalRows)} | Range ${formatCurrency(carrier.minFare)} to ${formatCurrency(carrier.maxFare)} | Bins ${formatNumber(carrier.bins.length)} -> ${formatNumber(consolidatedBins.length)}`;
          return (
            <HistogramCard
              key={carrier.carrier}
              title={`${carrier.carrierName} (${carrier.carrier})`}
              subtitle={subtitle}
              color={color}
              values={[]}
              buckets={consolidatedBins.map((bin) => ({
                label: `${formatCurrency(bin.fareStart)}-${formatCurrency(bin.fareEnd)}`,
                count: bin.passengers,
                tooltip: `${carrier.carrierName} (${carrier.carrier})\n${formatCurrency(bin.fareStart)}-${formatCurrency(bin.fareEnd)}\nPassengers: ${formatNumber(bin.passengers)}\nEntries: ${formatNumber(bin.rowCount)}${bin.sourceBinCount > 1 ? `\nMerged bins: ${formatNumber(bin.sourceBinCount)}` : ""}`,
              }))}
            />
          );
        })}
      </section>
    );
  }

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

      {routeSpecificScope ? (
        renderRouteSpecificFareDistribution()
      ) : (
          <HistogramCard
            title="Fare Distribution"
            subtitle={`${formatLocationSelectionLabel(originSelection, "All Origins")} -> ${formatLocationSelectionLabel(destSelection, "All Destinations")}`}
            values={[]}
            points={farePoints}
            bucketCount={standardFareBucketCount}
          />
        )}

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





