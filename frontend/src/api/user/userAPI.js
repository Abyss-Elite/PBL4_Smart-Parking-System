import api from "@/lib/axiosInstance";

const userAPI = {
  getUser: () => {
    return api.get("user/getInfo");
  },
  getUsers: () => {
    return api.get("user/getAllUsers");
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

export default userAPI;
