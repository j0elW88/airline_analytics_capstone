/**
 * @file src/utils/locationTaxonomy.ts
 * @description Location taxonomy and parsing helpers for All/Area/State/Airport filtering.
 */

import { AIRPORT_STATE_MAP } from "./airportStateMap";
import { getAirportDisplayName, normalizeAirportCode } from "./airports";

export type LocationScopeType = "all" | "area" | "state" | "airport";

export interface ParsedLocationSelection {
  type: LocationScopeType;
  code: string;
}

export interface LocationSelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

const STATE_NAMES: Record<string, string> = {
  AL: "Alabama",
  AK: "Alaska",
  AZ: "Arizona",
  AR: "Arkansas",
  CA: "California",
  CO: "Colorado",
  CT: "Connecticut",
  DE: "Delaware",
  FL: "Florida",
  GA: "Georgia",
  HI: "Hawaii",
  ID: "Idaho",
  IL: "Illinois",
  IN: "Indiana",
  IA: "Iowa",
  KS: "Kansas",
  KY: "Kentucky",
  LA: "Louisiana",
  ME: "Maine",
  MD: "Maryland",
  MA: "Massachusetts",
  MI: "Michigan",
  MN: "Minnesota",
  MS: "Mississippi",
  MO: "Missouri",
  MT: "Montana",
  NE: "Nebraska",
  NV: "Nevada",
  NH: "New Hampshire",
  NJ: "New Jersey",
  NM: "New Mexico",
  NY: "New York",
  NC: "North Carolina",
  ND: "North Dakota",
  OH: "Ohio",
  OK: "Oklahoma",
  OR: "Oregon",
  PA: "Pennsylvania",
  RI: "Rhode Island",
  SC: "South Carolina",
  SD: "South Dakota",
  TN: "Tennessee",
  TX: "Texas",
  UT: "Utah",
  VT: "Vermont",
  VA: "Virginia",
  WA: "Washington",
  WV: "West Virginia",
  WI: "Wisconsin",
  WY: "Wyoming",
  DC: "District of Columbia",
  PR: "Puerto Rico",
  VI: "U.S. Virgin Islands",
  GU: "Guam",
  AS: "American Samoa",
  MP: "Northern Mariana Islands",
  TT: "Trust Territories",
};

interface AreaDefinition {
  code: string;
  label: string;
  airports: string[];
}

// Proximity-based market areas where airports often serve overlapping demand.
const AREA_DEFINITIONS: AreaDefinition[] = [
  { code: "NYC_METRO", label: "NYC Metro", airports: ["JFK", "LGA", "EWR", "ISP", "HPN", "SWF", "TTN"] },
  { code: "DC_BALT", label: "DC-Baltimore", airports: ["DCA", "IAD", "BWI"] },
  { code: "CHICAGO_AREA", label: "Chicago Area", airports: ["ORD", "MDW"] },
  { code: "LOS_ANGELES_BASIN", label: "Los Angeles Basin", airports: ["LAX", "BUR", "LGB", "ONT", "SNA"] },
  { code: "BAY_AREA", label: "San Francisco Bay Area", airports: ["SFO", "OAK", "SJC"] },
  { code: "DALLAS_FT_WORTH", label: "Dallas-Fort Worth", airports: ["DFW", "DAL"] },
  { code: "HOUSTON_AREA", label: "Houston Area", airports: ["IAH", "HOU"] },
  { code: "SOUTH_FLORIDA", label: "South Florida", airports: ["MIA", "FLL", "PBI", "EYW"] },
  { code: "CENTRAL_FLORIDA", label: "Central Florida", airports: ["MCO", "TPA", "PIE", "SFB", "SRQ", "MLB"] },
  { code: "NORTH_FLORIDA", label: "North Florida", airports: ["JAX", "TLH", "GNV", "PNS", "VPS", "ECP"] },
  { code: "SEATTLE_AREA", label: "Seattle Area", airports: ["SEA", "PAE"] },
];

const AREA_LABELS: Record<string, string> = AREA_DEFINITIONS.reduce((acc, area) => {
  acc[area.code] = area.label;
  return acc;
}, {} as Record<string, string>);

const AIRPORT_TO_AREA_CODE: Record<string, string> = AREA_DEFINITIONS.reduce((acc, area) => {
  area.airports.forEach((airport) => {
    acc[airport] = area.code;
  });
  return acc;
}, {} as Record<string, string>);

const STATE_SORT_ORDER = [
  "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN",
  "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ",
  "NM", "NV", "NY", "OH", "OK", "OR", "PA", "PR", "RI", "SC", "SD", "TN", "TT", "TX", "UT", "VA",
  "VI", "VT", "WA", "WI", "WV", "WY",
];

const STATE_SORT_RANK: Record<string, number> = STATE_SORT_ORDER.reduce((acc, code, index) => {
  acc[code] = index;
  return acc;
}, {} as Record<string, number>);

const AREA_SORT_RANK: Record<string, number> = AREA_DEFINITIONS.reduce((acc, area, index) => {
  acc[area.code] = index;
  return acc;
}, {} as Record<string, number>);

function normalizeSelectionCode(code: string): string {
  return String(code || "").trim().toUpperCase();
}

function sortStates(codes: string[]): string[] {
  return [...codes].sort((a, b) => {
    const rankA = STATE_SORT_RANK[a] ?? Number.MAX_SAFE_INTEGER;
    const rankB = STATE_SORT_RANK[b] ?? Number.MAX_SAFE_INTEGER;
    if (rankA !== rankB) {
      return rankA - rankB;
    }
    return getStateDisplayName(a).localeCompare(getStateDisplayName(b));
  });
}

function sortAreas(codes: string[]): string[] {
  return [...codes].sort((a, b) => {
    const rankA = AREA_SORT_RANK[a] ?? Number.MAX_SAFE_INTEGER;
    const rankB = AREA_SORT_RANK[b] ?? Number.MAX_SAFE_INTEGER;
    if (rankA !== rankB) {
      return rankA - rankB;
    }
    return getAreaDisplayName(a).localeCompare(getAreaDisplayName(b));
  });
}

function sortAirports(codes: string[]): string[] {
  return [...codes].sort((a, b) => getAirportDisplayName(a).localeCompare(getAirportDisplayName(b)));
}

export function parseLocationSelection(raw: string): ParsedLocationSelection {
  const value = String(raw || "").trim();
  if (!value) {
    return { type: "all", code: "" };
  }
  const upper = value.toUpperCase();
  if (upper.startsWith("AIRPORT:")) {
    return { type: "airport", code: normalizeAirportCode(upper.slice("AIRPORT:".length)) };
  }
  if (upper.startsWith("STATE:")) {
    return { type: "state", code: normalizeSelectionCode(upper.slice("STATE:".length)) };
  }
  if (upper.startsWith("AREA:")) {
    return { type: "area", code: normalizeSelectionCode(upper.slice("AREA:".length)) };
  }
  // Backward compatibility with older plain airport-code values.
  return { type: "airport", code: normalizeAirportCode(upper) };
}

export function toLocationSelectionValue(selection: ParsedLocationSelection): string {
  if (selection.type === "all" || !selection.code) {
    return "";
  }
  if (selection.type === "airport") {
    return `AIRPORT:${selection.code}`;
  }
  if (selection.type === "state") {
    return `STATE:${selection.code}`;
  }
  return `AREA:${selection.code}`;
}

export function normalizeLocationSelectionValue(raw: string): string {
  return toLocationSelectionValue(parseLocationSelection(raw));
}

export function getAirportStateCode(airportCode: string): string {
  const code = normalizeAirportCode(airportCode);
  return AIRPORT_STATE_MAP[code] ?? "";
}

export function getAirportAreaCode(airportCode: string): string {
  const code = normalizeAirportCode(airportCode);
  return AIRPORT_TO_AREA_CODE[code] ?? "";
}

export function getStateDisplayName(stateCode: string): string {
  const code = normalizeSelectionCode(stateCode);
  const name = STATE_NAMES[code] ?? "Unknown State";
  return `${code} - ${name}`;
}

export function getAreaDisplayName(areaCode: string): string {
  const code = normalizeSelectionCode(areaCode);
  return AREA_LABELS[code] ?? code;
}

export function formatLocationSelectionLabel(selection: ParsedLocationSelection, allLabel = "All"): string {
  if (selection.type === "all") {
    return allLabel;
  }
  if (selection.type === "airport") {
    return getAirportDisplayName(selection.code);
  }
  if (selection.type === "state") {
    return `State - ${getStateDisplayName(selection.code)}`;
  }
  return `Area - ${getAreaDisplayName(selection.code)}`;
}

export function matchesAirportToSelection(airportCode: string, selection: ParsedLocationSelection): boolean {
  if (selection.type === "all") {
    return true;
  }
  const airport = normalizeAirportCode(airportCode);
  if (!airport) {
    return false;
  }
  if (selection.type === "airport") {
    return airport === selection.code;
  }
  if (selection.type === "state") {
    return getAirportStateCode(airport) === selection.code;
  }
  return getAirportAreaCode(airport) === selection.code;
}

export function buildLocationSelectOptions(airportCodes: string[], allLabel: string): LocationSelectOption[] {
  const normalizedAirports = Array.from(new Set(
    airportCodes.map((code) => normalizeAirportCode(code)).filter((code) => code.length === 3),
  ));
  const options: LocationSelectOption[] = [{ value: "", label: allLabel }];

  const areaAirportMap = new Map<string, string[]>();
  normalizedAirports.forEach((airport) => {
    const area = getAirportAreaCode(airport);
    if (!area) {
      return;
    }
    if (!areaAirportMap.has(area)) {
      areaAirportMap.set(area, []);
    }
    (areaAirportMap.get(area) as string[]).push(airport);
  });

  const byState = new Map<string, string[]>();
  normalizedAirports.forEach((airport) => {
    const stateCode = getAirportStateCode(airport) || "UNK";
    if (!byState.has(stateCode)) {
      byState.set(stateCode, []);
    }
    (byState.get(stateCode) as string[]).push(airport);
  });

  sortStates(Array.from(byState.keys())).forEach((stateCode) => {
    if (stateCode !== "UNK") {
      options.push({
        value: `STATE:${stateCode}`,
        label: `State - ${getStateDisplayName(stateCode)}`,
      });
    }
    sortAirports(byState.get(stateCode) ?? []).forEach((airport) => {
      options.push({
        value: `AIRPORT:${airport}`,
        label: `\u00A0\u00A0${getAirportDisplayName(airport)}`,
      });
    });
  });

  const sortedAreas = sortAreas(Array.from(areaAirportMap.keys()));
  if (sortedAreas.length > 0) {
    options.push({
      value: "__AREA_HEADER__",
      label: "Areas",
      disabled: true,
    });
    sortedAreas.forEach((areaCode) => {
      options.push({
        value: `AREA:${areaCode}`,
        label: `\u00A0\u00A0Area - ${getAreaDisplayName(areaCode)}`,
      });
    });
  }

  return options;
}
