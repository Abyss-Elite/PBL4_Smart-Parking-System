import api from "@/lib/axiosInstance";

const userAPI = {
  getUser: () => {
    return api.get("user/getInfo");
  }
  getUsers: () => {
    return api.get("user/getAllUsers");
  },
};
export default userAPI;
