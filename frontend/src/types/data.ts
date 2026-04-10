/**
 * @file src/types/data.ts
 * @description Shared TypeScript data contracts used across state, services, and UI.
 */

export type Screen =
  | "home"
  | "history"
  | "loaded"
  | "start"
  | "load"
  | "help"
  | "about"
  | "analyze_one"
  | "analyze_multi"
  | "results_one"
  | "results_multi";

export type PeriodKey = `${number}_Q${1 | 2 | 3 | 4}`;

export interface RouteMarketPowerRow {
  Origin: string;
  Dest: string;
  Carrier: string;
  carrier_name?: string;
  total_passengers: number;
  row_count?: number;
  avg_fare_weighted: number;
  avg_distance_weighted: number;
  route_total_passengers_all?: number;
  route_total_passengers_valid?: number;
  carriers_on_route_all?: number;
  carriers_on_route_valid?: number;
  route_share?: number;
  route_HHI?: number;
  route_avg_fare_all?: number;
  route_min_fare_all?: number;
  OriginState?: string;
}

export interface HubMarketPowerRow {
  Origin: string;
  OriginState: string;
  Carrier: string;
  carrier_name?: string;
  total_passengers: number;
  row_count?: number;
  avg_fare_weighted: number;
  avg_distance_weighted: number;
  hub_total_passengers_all?: number;
  hub_total_passengers_valid?: number;
  carriers_at_hub_all?: number;
  carriers_at_hub_valid?: number;
  hub_share?: number;
  hub_HHI?: number;
  hub_avg_fare_all?: number;
  hub_min_fare_all?: number;
}

export interface DatasetRecord {
  period: PeriodKey;
  routeRows: RouteMarketPowerRow[];
  hubRows: HubMarketPowerRow[];
  uploadedAtIso: string;
}

export interface RouteFilters {
  origin: string;
  dest: string;
  carrier: string;
}

export interface HubFilters {
  carrier: string;
}

export interface ModalAction {
  label: string;
  kind?: "default" | "primary" | "danger";
  onClick?: () => void;
}

export interface ModalConfig {
  title: string;
  message?: string;
  content?: string;
  actions?: ModalAction[];
}

export interface AppState {
  screen: Screen;
  stack: Screen[];
  history: string[];
  datasetsByPeriod: Record<string, DatasetRecord>;
  selectedSinglePeriod: string | null;
  selectedMultiPeriods: string[];
  modal: ModalConfig | null;
}





