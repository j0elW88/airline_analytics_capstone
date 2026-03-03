/**
 * @file src/hooks/useCarrierLookup.ts
 * @description Hook that loads and caches carrier code-to-name lookups.
 */

import { useEffect, useState } from "react";
import { fetchCarrierLookup } from "../services/localBackend";

let carrierLookupCache: Record<string, string> | null = null;
let carrierLookupPromise: Promise<Record<string, string>> | null = null;

async function loadCarrierLookup(): Promise<Record<string, string>> {
  if (carrierLookupCache) {
    return carrierLookupCache;
  }
  if (carrierLookupPromise) {
    return carrierLookupPromise;
  }

  carrierLookupPromise = fetchCarrierLookup()
    .then((lookup) => {
      carrierLookupCache = lookup;
      return lookup;
    })
    .catch(() => ({}))
    .finally(() => {
      carrierLookupPromise = null;
    });

  return carrierLookupPromise;
}

export function useCarrierLookup(): Record<string, string> {
  const [lookup, setLookup] = useState<Record<string, string>>(carrierLookupCache ?? {});

  useEffect(() => {
    let active = true;
    void loadCarrierLookup().then((loaded) => {
      if (active) {
        setLookup(loaded);
      }
    });
    return () => {
      active = false;
    };
  }, []);

  return lookup;
}






