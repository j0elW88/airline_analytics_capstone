/**
 * @file src/features/results/analytics.ts
 * @description Pure data transformation and aggregation helpers for analytics views.
 */

import type {
  HubFilters,
  HubMarketPowerRow,
  RouteFilters,
  RouteMarketPowerRow,
} from "../../types/data";
import { normalizeCarrierCode } from "../../utils/carrierDisplay";
import { formatRouteDisplay } from "../../utils/airports";
import { roundToNearest } from "../../utils/format";

export function applyRouteFilters(rows: RouteMarketPowerRow[], filters: RouteFilters): RouteMarketPowerRow[] {
  // Applies origin/destination/carrier constraints to route rows.
  return rows.filter((row) => {
    if (!row) {
      return false;
    }
    const carrierCode = normalizeCarrierCode(row.Carrier);
    if (filters.origin && row.Origin !== filters.origin) {
      return false;
    }
    if (filters.dest && row.Dest !== filters.dest) {
      return false;
    }
    if (filters.carrier && carrierCode !== normalizeCarrierCode(filters.carrier)) {
      return false;
    }
    return true;
  });
}

export function applyHubFilters(rows: HubMarketPowerRow[], filters: HubFilters): HubMarketPowerRow[] {
  // Applies carrier constraint to hub rows.
  return rows.filter((row) => {
    if (!row) {
      return false;
    }
    const carrierCode = normalizeCarrierCode(row.Carrier);
    if (filters.carrier && carrierCode !== normalizeCarrierCode(filters.carrier)) {
      return false;
    }
    return true;
  });
}

export interface MarketOverviewStats {
  totalPassengers: number;
  avgFare: number;
  carriers: number;
  avgHhi: number;
}

export function computeMarketOverview(rows: RouteMarketPowerRow[]): MarketOverviewStats {
  // Computes top-level KPIs for market overview cards.
  if (rows.length === 0) {
    return {
      totalPassengers: 0,
      avgFare: 0,
      carriers: 0,
      avgHhi: 0,
    };
  }

  const totalPassengers = rows.reduce((sum, row) => sum + row.total_passengers, 0);
  const weightedFareSum = rows.reduce((sum, row) => sum + row.avg_fare_weighted * row.total_passengers, 0);
  const avgFare = totalPassengers > 0 ? weightedFareSum / totalPassengers : 0;

  const carrierSet = new Set(rows.map((row) => normalizeCarrierCode(row.Carrier)));
  const routeHhiMap = new Map<string, number>();
  rows.forEach((row) => {
    const key = `${row.Origin}_${row.Dest}`;
    if (!routeHhiMap.has(key) && row.route_HHI) {
      routeHhiMap.set(key, row.route_HHI);
    }
  });

  const hhiValues = Array.from(routeHhiMap.values());
  const avgHhi = hhiValues.length > 0
    ? hhiValues.reduce((sum, value) => sum + value, 0) / hhiValues.length
    : 0;

  return {
    totalPassengers,
    avgFare,
    carriers: carrierSet.size,
    avgHhi,
  };
}

export interface TopRouteRow {
  route: string;
  passengers: number;
}

export function getTopRoutes(rows: RouteMarketPowerRow[], limit = 10): TopRouteRow[] {
  // Ranks routes by total observed passengers.
  const byRoute = new Map<string, number>();
  rows.forEach((row) => {
    if (!row) {
      return;
    }
    const key = formatRouteDisplay(row.Origin, row.Dest);
    byRoute.set(key, (byRoute.get(key) ?? 0) + row.total_passengers);
  });

  return Array.from(byRoute.entries())
    .map(([route, passengers]) => ({ route, passengers }))
    .sort((a, b) => b.passengers - a.passengers)
    .slice(0, limit);
}

export interface CarrierSummary {
  carrier: string;
  passengers: number;
  avgFare: number;
  revenueProxy: number;
  totalMileage: number;
  usPassengerShare: number;
  estimatedFlights: number;
}

export function summarizeByCarrier(rows: RouteMarketPowerRow[]): CarrierSummary[] {
  // Builds carrier-level aggregates used by summary tables and bar charts.
  const totalPassengers = rows.reduce((sum, row) => sum + row.total_passengers, 0);
  const map = new Map<string, { passengers: number; fareXPass: number; mileage: number }>();

  rows.forEach((row) => {
    if (!row) {
      return;
    }
    const key = normalizeCarrierCode(row.Carrier);
    const existing = map.get(key) ?? { passengers: 0, fareXPass: 0, mileage: 0 };
    existing.passengers += row.total_passengers;
    existing.fareXPass += row.avg_fare_weighted * row.total_passengers;
    existing.mileage += row.avg_distance_weighted * row.total_passengers;
    map.set(key, existing);
  });

  return Array.from(map.entries())
    .map(([carrier, data]) => {
      const avgFare = data.passengers > 0 ? data.fareXPass / data.passengers : 0;
      return {
        carrier,
        passengers: data.passengers,
        avgFare,
        revenueProxy: data.fareXPass,
        totalMileage: data.mileage,
        usPassengerShare: totalPassengers > 0 ? data.passengers / totalPassengers : 0,
        estimatedFlights: roundToNearest(data.passengers * 10, 1000),
      };
    })
    .sort((a, b) => b.passengers - a.passengers);
}

export function getHighCostRoutes(rows: RouteMarketPowerRow[], limit = 10): Array<{ route: string; avgFare: number }> {
  // Ranks routes by weighted average fare.
  const map = new Map<string, { fareXPass: number; passengers: number }>();
  rows.forEach((row) => {
    if (!row) {
      return;
    }
    const key = formatRouteDisplay(row.Origin, row.Dest);
    const current = map.get(key) ?? { fareXPass: 0, passengers: 0 };
    current.fareXPass += row.avg_fare_weighted * row.total_passengers;
    current.passengers += row.total_passengers;
    map.set(key, current);
  });

  return Array.from(map.entries())
    .map(([route, values]) => ({
      route,
      avgFare: values.passengers > 0 ? values.fareXPass / values.passengers : 0,
    }))
    .sort((a, b) => b.avgFare - a.avgFare)
    .slice(0, limit);
}

export function getCarrierShareBars(rows: RouteMarketPowerRow[], limit?: number): Array<{ label: string; value: number }> {
  // Converts carrier summaries into percent-share bar rows.
  const summaries = summarizeByCarrier(rows);
  const totalPassengers = summaries.reduce((sum, row) => sum + row.passengers, 0);
  const visible = typeof limit === "number" ? summaries.slice(0, limit) : summaries;
  return visible.map((row) => ({
    label: row.carrier,
    value: totalPassengers > 0 ? (row.passengers / totalPassengers) * 100 : 0,
  }));
}

export function getCarrierFareBars(rows: RouteMarketPowerRow[], limit?: number): Array<{ label: string; value: number }> {
  // Converts carrier summaries into average-fare bar rows.
  const summaries = summarizeByCarrier(rows);
  const visible = typeof limit === "number" ? summaries.slice(0, limit) : summaries;
  return visible
    .map((item) => ({
      label: item.carrier,
      value: item.avgFare,
    }));
}

export function getFareValues(rows: RouteMarketPowerRow[]): number[] {
  // Extracts valid fare numbers for histogram rendering.
  return rows
    .filter(Boolean)
    .map((row) => row.avg_fare_weighted)
    .filter((value) => Number.isFinite(value) && value > 0);
}

export interface FareDistributionPoint {
  value: number;
  carrier: string;
  weight: number;
}

export function getFareDistributionPoints(rows: RouteMarketPowerRow[]): FareDistributionPoint[] {
  // Keeps fare points linked to carrier + passengers so histogram bins can explain contributors.
  return rows
    .filter(Boolean)
    .map((row) => ({
      value: row.avg_fare_weighted,
      carrier: normalizeCarrierCode(row.Carrier),
      weight: Number.isFinite(row.total_passengers) && row.total_passengers > 0 ? row.total_passengers : 1,
    }))
    .filter((point) => Number.isFinite(point.value) && point.value > 0 && point.carrier.length > 0);
}

export type SelectionMode =
  | "all"
  | "origin"
  | "dest"
  | "carrier"
  | "origin_dest"
  | "origin_dest_carrier";

export function detectSelectionMode(filters: RouteFilters): SelectionMode {
  // Maps active filters to a display mode key.
  if (filters.origin && filters.dest && filters.carrier) {
    return "origin_dest_carrier";
  }
  if (filters.origin && filters.dest) {
    return "origin_dest";
  }
  if (filters.carrier) {
    return "carrier";
  }
  if (filters.origin) {
    return "origin";
  }
  if (filters.dest) {
    return "dest";
  }
  return "all";
}

export function getSelectionModeTitle(mode: SelectionMode): string {
  // Friendly labels for the current filter/display mode.
  switch (mode) {
    case "origin":
      return "Origin-focused display";
    case "dest":
      return "Destination-focused display";
    case "carrier":
      return "Carrier-focused display";
    case "origin_dest":
      return "Route market display";
    case "origin_dest_carrier":
      return "Route + carrier deep dive";
    case "all":
    default:
      return "National overview display";
  }
}

export interface RouteMarketSnapshot {
  route: string;
  carrierCount: number;
  carriers: string[];
  avgFare: number;
  avgSharePct: number;
  fareGapVsMin: number;
  passengers: number;
}

export function getRouteMarketSnapshot(rows: RouteMarketPowerRow[], limit = 12): RouteMarketSnapshot[] {
  // Produces per-route competitiveness and pricing metrics.
  const byRoute = new Map<string, {
    fareXPass: number;
    passengers: number;
    minFare: number;
    carriers: Set<string>;
    shareSum: number;
    shareCount: number;
  }>();

  rows.forEach((row) => {
    if (!row) {
      return;
    }
    const key = formatRouteDisplay(row.Origin, row.Dest);
    const existing = byRoute.get(key) ?? {
      fareXPass: 0,
      passengers: 0,
      minFare: Number.POSITIVE_INFINITY,
      carriers: new Set<string>(),
      shareSum: 0,
      shareCount: 0,
    };

    existing.fareXPass += row.avg_fare_weighted * row.total_passengers;
    existing.passengers += row.total_passengers;
    existing.minFare = Math.min(existing.minFare, row.route_min_fare_all || row.avg_fare_weighted);
    existing.carriers.add(normalizeCarrierCode(row.Carrier));

    if (row.route_share && row.route_share > 0) {
      existing.shareSum += row.route_share;
      existing.shareCount += 1;
    }

    byRoute.set(key, existing);
  });

  return Array.from(byRoute.entries())
    .map(([route, values]) => {
      const avgFare = values.passengers > 0 ? values.fareXPass / values.passengers : 0;
      const avgShare = values.shareCount > 0
        ? (values.shareSum / values.shareCount) * 100
        : (values.carriers.size > 0 ? 100 / values.carriers.size : 0);
      return {
        route,
        carrierCount: values.carriers.size,
        carriers: Array.from(values.carriers).sort(),
        avgFare,
        avgSharePct: avgShare,
        fareGapVsMin: avgFare - (Number.isFinite(values.minFare) ? values.minFare : avgFare),
        passengers: values.passengers,
      };
    })
    .sort((a, b) => b.passengers - a.passengers)
    .slice(0, limit);
}

export function getCarrierRouteBreakdown(rows: RouteMarketPowerRow[], limit = 12): Array<{
  route: string;
  passengers: number;
  avgFare: number;
}> {
  // Produces route stats for whichever carrier scope is currently selected.
  const map = new Map<string, { passengers: number; fareXPass: number }>();

  rows.forEach((row) => {
    if (!row) {
      return;
    }
    const key = formatRouteDisplay(row.Origin, row.Dest);
    const current = map.get(key) ?? { passengers: 0, fareXPass: 0 };
    current.passengers += row.total_passengers;
    current.fareXPass += row.avg_fare_weighted * row.total_passengers;
    map.set(key, current);
  });

  return Array.from(map.entries())
    .map(([route, values]) => ({
      route,
      passengers: values.passengers,
      avgFare: values.passengers > 0 ? values.fareXPass / values.passengers : 0,
    }))
    .sort((a, b) => b.passengers - a.passengers)
    .slice(0, limit);
}

export function summarizeHubMarkets(
  rows: HubMarketPowerRow[],
  routeRows: RouteMarketPowerRow[] = [],
): Array<{
  hub: string;
  passengers: number;
  avgFare: number;
  destinationsServed: number;
}> {
  // Produces hub-level passenger totals, fares, and number of destinations served.
  const map = new Map<string, { passengers: number; fareXPass: number }>();

  rows.forEach((row) => {
    if (!row) {
      return;
    }
    const key = `${row.Origin} (${row.OriginState})`;
    const current = map.get(key) ?? { passengers: 0, fareXPass: 0 };
    current.passengers += row.total_passengers;
    current.fareXPass += row.avg_fare_weighted * row.total_passengers;
    map.set(key, current);
  });

  const destinationsByOrigin = new Map<string, Set<string>>();
  routeRows.forEach((row) => {
    if (!row) {
      return;
    }
    const set = destinationsByOrigin.get(row.Origin) ?? new Set<string>();
    set.add(row.Dest);
    destinationsByOrigin.set(row.Origin, set);
  });

  return Array.from(map.entries())
    .map(([hub, values]) => ({
      hub,
      passengers: values.passengers,
      avgFare: values.passengers ? values.fareXPass / values.passengers : 0,
      destinationsServed: destinationsByOrigin.get(hub.split(" (")[0])?.size ?? 0,
    }))
    .sort((a, b) => b.passengers - a.passengers);
}

export function getHubPassengerBars(
  hubRows: HubMarketPowerRow[],
  routeRows: RouteMarketPowerRow[],
  limit = 10,
): Array<{ label: string; value: number }> {
  // Converts hub summaries into passenger bars for chart visualization.
  return summarizeHubMarkets(hubRows, routeRows)
    .slice(0, limit)
    .map((row) => ({
      label: row.hub,
      value: row.passengers,
    }));
}





