import axios from "axios";
import { CONFIG } from "./config";
import { getAccessToken, getRefreshToken, setToken, removeToken } from "@/utils/tokenStorage";

const api = axios.create({
  baseURL: CONFIG.API_BASE_URL + "/",
});

api.interceptors.request.use((config) => {
  const accessToken = getAccessToken();
  if (accessToken) {
    config.headers["Authorization"] = `Bearer ${accessToken}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  async (err) => {
    const originalRequest = err.config;
    if (err.response?.status === 403) {
      try {
        const refreshTokenOld = getRefreshToken();
        if (!refreshTokenOld) throw new Error("refresh token not available");

        const res = await api.post(`${CONFIG.API_BASE_URL}/user/refreshToken`, {
          refreshToken: refreshTokenOld,
        });

        const { accessToken, refreshToken } = res.data.result;
        setToken({ accessToken: accessToken, refreshToken: refreshToken });
        originalRequest.headers["Authorization"] = `Bearer ${accessToken}`;
        return api(originalRequest);
      } catch (refreshError) {
        console.log("Refresh token expired - redirecting to login");
        // removeToken();
        // window.location.href = PATH.LOGIN;
        return Promise.reject(refreshError);
      }
    }
  }
);

export default api;
