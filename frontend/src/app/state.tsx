/**
 * @file src/app/state.tsx
 * @description Global app state container with reducer actions, persistence, and derived selectors.
 */

import { createContext, useContext, useEffect, useMemo, useReducer } from "react";
import type {
  AppState,
  DatasetRecord,
  ModalConfig,
  Screen,
} from "../types/data";

const STORAGE_KEY = "airline_analytics_frontend_state_v1";

interface PersistedAppState {
  screen: Screen;
  stack: Screen[];
  history: string[];
  selectedSinglePeriod: string | null;
  selectedMultiPeriods: string[];
}

const initialState: AppState = {
  // App always starts at the launch screen on full refresh.
  screen: "start",
  stack: [],
  history: [],
  datasetsByPeriod: {},
  selectedSinglePeriod: null,
  selectedMultiPeriods: [],
  modal: null,
};

type Action =
  | { type: "NAV_TO"; screen: Screen }
  | { type: "NAV_BACK" }
  | { type: "ADD_HISTORY"; item: string }
  | { type: "UPSERT_DATASET"; dataset: DatasetRecord }
  | { type: "SET_SINGLE_PERIOD"; period: string | null }
  | { type: "SET_MULTI_PERIODS"; periods: string[] }
  | { type: "OPEN_MODAL"; modal: ModalConfig }
  | { type: "CLOSE_MODAL" }
  | { type: "RESET" };

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "NAV_TO": {
      // Avoid pointless stack growth when user clicks current screen.
      if (state.screen === action.screen) {
        return state;
      }
      return {
        ...state,
        stack: [...state.stack, state.screen],
        screen: action.screen,
      };
    }
    case "NAV_BACK": {
      // If stack is empty, fall back to start page instead of crashing.
      if (state.stack.length === 0) {
        return { ...state, screen: "start" };
      }
      const next = [...state.stack];
      const previousScreen = next.pop() ?? "home";
      return {
        ...state,
        stack: next,
        screen: previousScreen,
      };
    }
    case "ADD_HISTORY": {
      // Keep a bounded history list to avoid unbounded localStorage growth.
      const history = [action.item, ...state.history].slice(0, 50);
      return { ...state, history };
    }
    case "UPSERT_DATASET": {
      return {
        ...state,
        datasetsByPeriod: {
          ...state.datasetsByPeriod,
          [action.dataset.period]: action.dataset,
        },
      };
    }
    case "SET_SINGLE_PERIOD": {
      return { ...state, selectedSinglePeriod: action.period };
    }
    case "SET_MULTI_PERIODS": {
      return { ...state, selectedMultiPeriods: action.periods };
    }
    case "OPEN_MODAL": {
      return { ...state, modal: action.modal };
    }
    case "CLOSE_MODAL": {
      return { ...state, modal: null };
    }
    case "RESET": {
      return initialState;
    }
    default:
      return state;
  }
}

function loadInitialState(): AppState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return initialState;
    }
    const parsed = JSON.parse(raw) as Partial<PersistedAppState>;
    return {
      ...initialState,
      history: Array.isArray(parsed.history) ? parsed.history : [],
      selectedSinglePeriod: typeof parsed.selectedSinglePeriod === "string" ? parsed.selectedSinglePeriod : null,
      selectedMultiPeriods: Array.isArray(parsed.selectedMultiPeriods) ? parsed.selectedMultiPeriods : [],
      // Always reset screen/stack/modal on fresh browser load.
      screen: "start",
      stack: [],
      modal: null,
    };
  } catch {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignore storage cleanup errors.
    }
    return initialState;
  }
}

interface AppStateContextValue {
  state: AppState;
  navTo: (screen: Screen) => void;
  navBack: () => void;
  addHistory: (item: string) => void;
  upsertDataset: (dataset: DatasetRecord) => void;
  setSinglePeriod: (period: string | null) => void;
  setMultiPeriods: (periods: string[]) => void;
  openModal: (modal: ModalConfig) => void;
  closeModal: () => void;
  resetState: () => void;
}

const AppStateContext = createContext<AppStateContextValue | null>(null);

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState, loadInitialState);

  useEffect(() => {
    // Persist only safe UI state (not heavy dataset rows) to keep storage small.
    const persistable: PersistedAppState = {
      screen: "start",
      stack: [],
      history: state.history,
      selectedSinglePeriod: state.selectedSinglePeriod,
      selectedMultiPeriods: state.selectedMultiPeriods,
    };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(persistable));
    } catch {
      // Ignore quota errors to prevent UI crash.
    }
  }, [state]);

  const value = useMemo<AppStateContextValue>(
    () => ({
      state,
      // Expose typed action wrappers so pages/components never dispatch raw action objects.
      navTo: (screen) => dispatch({ type: "NAV_TO", screen }),
      navBack: () => dispatch({ type: "NAV_BACK" }),
      addHistory: (item) => dispatch({ type: "ADD_HISTORY", item }),
      upsertDataset: (dataset) => dispatch({ type: "UPSERT_DATASET", dataset }),
      setSinglePeriod: (period) => dispatch({ type: "SET_SINGLE_PERIOD", period }),
      setMultiPeriods: (periods) => dispatch({ type: "SET_MULTI_PERIODS", periods }),
      openModal: (modal) => dispatch({ type: "OPEN_MODAL", modal }),
      closeModal: () => dispatch({ type: "CLOSE_MODAL" }),
      resetState: () => dispatch({ type: "RESET" }),
    }),
    [state],
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const context = useContext(AppStateContext);
  if (!context) {
    throw new Error("useAppState must be used within AppStateProvider");
  }
  return context;
}

export function getSortedPeriods(state: AppState): string[] {
  // Sort period keys once in selector so pages can consume in display order.
  return Object.keys(state.datasetsByPeriod).sort();
}

export function getCompletePeriods(state: AppState): string[] {
  // A period is "complete" only when both route and hub files are present.
  return getSortedPeriods(state).filter((period) => {
    const dataset = state.datasetsByPeriod[period];
    const routeRows = Array.isArray(dataset?.routeRows) ? dataset.routeRows : [];
    const hubRows = Array.isArray(dataset?.hubRows) ? dataset.hubRows : [];
    return Boolean(dataset) && routeRows.length > 0 && hubRows.length > 0;
  });
}





