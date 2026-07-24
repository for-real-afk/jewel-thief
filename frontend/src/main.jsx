import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./theme.css";
import App from "./App.jsx";
import CatalogUpload from "./CatalogUpload.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/catalog" element={<CatalogUpload />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
