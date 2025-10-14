const ACCESS_TOKEN = "accessToken";
const REFRESH_TOKEN = "refreshToken";

export const setToken = ({ accessToken, refreshToken }) => {
  if (accessToken) localStorage.setItem(ACCESS_TOKEN, accessToken);
  if (refreshToken) localStorage.setItem(REFRESH_TOKEN, refreshToken);
};


export const getAccessToken = () => {
  return localStorage.getItem(ACCESS_TOKEN);
};

export const getRefreshToken = () => {
  return localStorage.getItem(REFRESH_TOKEN);
};

export const removeToken = () => {
  localStorage.removeItem(ACCESS_TOKEN);
  localStorage.removeItem(REFRESH_TOKEN);
};
