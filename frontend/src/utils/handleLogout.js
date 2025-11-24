export const handleLogout = () => {
  if (typeof window !== "undefined") {
    localStorage.clear();
  }
  window.location.href = "/login";
};
