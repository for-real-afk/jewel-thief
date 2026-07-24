import { NavLink } from "react-router-dom";

export default function TopNav() {
  return (
    <nav className="top-nav">
      <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
        Search
      </NavLink>
      <NavLink to="/catalog" className={({ isActive }) => (isActive ? "active" : "")}>
        Catalog
      </NavLink>
    </nav>
  );
}
