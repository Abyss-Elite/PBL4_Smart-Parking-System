import api from "@/lib/axiosInstance";

const authAPI = {
  login: ({ email, password }) => {
    return api.post("/api/user/login", { email, password });
  },
  register: ({ username, email, password, role = "User" }) => {
    return api.post("/api/user/register", { username, email, password, role });
  },
  resetPassword: ({ username, email, password }) => {
    return api.post("/api/user/reset-password", { username, email, password });
  },
};

export default authAPI;
