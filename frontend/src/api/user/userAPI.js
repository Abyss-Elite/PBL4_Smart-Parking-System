import api from "@/lib/axiosInstance";

const userAPI = {
  getUsers: () => {
    return api.get("user/getAllUsers");
  },
};
export default userAPI;
