/**
 * @file src/main.tsx
 * @description Vite entrypoint that mounts the React app and global providers.
 */

import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./app/App";
import { AppStateProvider } from "./app/state";
import "./styles/theme.css";
import "./styles/app.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppStateProvider>
      <App />
    </AppStateProvider>
  </React.StrictMode>,
);





