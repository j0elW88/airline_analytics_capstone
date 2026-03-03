/**
 * @file src/components/filters/HubFilterBar.tsx
 * @description Hub-focused filter controls for period and carrier selection.
 */

import type { HubFilters, HubMarketPowerRow } from "../../types/data";
import { useCarrierLookup } from "../../hooks/useCarrierLookup";
import { getCarrierDisplayName, normalizeCarrierCode } from "../../utils/carrierDisplay";

interface HubFilterBarProps {
  filters: HubFilters;
  onChange: (next: HubFilters) => void;
  period: string;
  rows: HubMarketPowerRow[];
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean))).sort();
}

export function HubFilterBar({ filters, onChange, period, rows }: HubFilterBarProps) {
  const carrierLookup = useCarrierLookup();
  const [year, quarter] = period.split("_Q");
  const carrierCodes = unique(rows.map((row) => normalizeCarrierCode(row.Carrier)));

  return (
    <section className="filter-grid filter-grid--hub">
      <label>
        Year
        <input value={year || "-"} disabled />
      </label>
      <label>
        Quarter
        <input value={quarter ? `Q${quarter}` : "-"} disabled />
      </label>
      <label>
        Carrier
        <select
          value={filters.carrier}
          onChange={(event) => onChange({ carrier: event.target.value })}
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





