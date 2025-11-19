import api from "@/lib/axiosInstance";

export const authAPI = {
  login: ({ email, password }) => {
    return api.post("user/login", { email, password });
  },
  register: ({ username, email, password, role = "User" }) => {
    return api.post("user/register", { username, email, password, role });
  },
  resetPassword: ({ username, email, password }) => {
    return api.post("user/reset-password", { username, email, password });
  },
};
