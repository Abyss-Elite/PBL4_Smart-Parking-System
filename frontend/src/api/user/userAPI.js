import api from "@/lib/axiosInstance";

export const userAPI = {
  getUser: (userId) => {
    return api.get(`user/${userId}`);
  },
  getUsers: () => {
    return api.get("user");
  },
  getNumberUsers: () => {
    return api.get("user/userNumber");
  },
  createUser: (user) => {
    return api.post("user", user);
  },
  updateUser: (user) => {
    return api.put(`user/${user.id}`, user);
  },
  deleteUser: (id) => {
    return api.delete(`user/${id}`);
  },
};
