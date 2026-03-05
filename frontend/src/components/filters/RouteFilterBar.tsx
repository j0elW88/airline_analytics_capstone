/**
 * @file src/components/filters/RouteFilterBar.tsx
 * @description Route-focused filter controls for origin, destination, and carrier selection.
 */

import type { RouteFilters, RouteMarketPowerRow } from "../../types/data";
import { useCarrierLookup } from "../../hooks/useCarrierLookup";
import { getCarrierDisplayName, normalizeCarrierCode } from "../../utils/carrierDisplay";
import {
  buildLocationSelectOptions,
  normalizeLocationSelectionValue,
} from "../../utils/locationTaxonomy";

interface RouteFilterBarProps {
  filters: RouteFilters;
  onChange: (next: RouteFilters) => void;
  period: string;
  rows: RouteMarketPowerRow[];
  showOrigin?: boolean;
  showDestination?: boolean;
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean))).sort();
}

export function RouteFilterBar({
  filters,
  onChange,
  period,
  rows,
  showOrigin = true,
  showDestination = true,
}: RouteFilterBarProps) {
  const carrierLookup = useCarrierLookup();
  const [year, quarter] = period.split("_Q");

  const origins = unique(rows.map((row) => row.Origin));
  const dests = unique(rows.map((row) => row.Dest));
  const carrierCodes = unique(rows.map((row) => normalizeCarrierCode(row.Carrier)));
  const originOptions = buildLocationSelectOptions(origins, "All Origins");
  const destinationOptions = buildLocationSelectOptions(dests, "All Destinations");
  const selectedOriginValue = normalizeLocationSelectionValue(filters.origin);
  const selectedDestinationValue = normalizeLocationSelectionValue(filters.dest);

  return (
    <section className="filter-grid">
      <label>
        Year
        <input value={year || "-"} disabled />
      </label>
      <label>
        Quarter
        <input value={quarter ? `Q${quarter}` : "-"} disabled />
      </label>
      {showOrigin ? (
        <label>
          Origin
          <select
            value={selectedOriginValue}
            onChange={(event) => onChange({ ...filters, origin: event.target.value })}
          >
            {originOptions.map((option) => (
              <option key={option.value || "__all_origin__"} value={option.value} disabled={option.disabled}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {showDestination ? (
        <label>
          Destination
          <select
            value={selectedDestinationValue}
            onChange={(event) => onChange({ ...filters, dest: event.target.value })}
          >
            {destinationOptions.map((option) => (
              <option key={option.value || "__all_dest__"} value={option.value} disabled={option.disabled}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      <label>
        Carrier
        <select
          value={filters.carrier}
          onChange={(event) => onChange({ ...filters, carrier: event.target.value })}
        >
          <option value="">All Carriers</option>
          {carrierCodes.map((code) => (
            <option key={code} value={code}>
              {getCarrierDisplayName(code, carrierLookup)}
            </option>
          ))}
        </select>
      </label>
    </section>
  );
}





