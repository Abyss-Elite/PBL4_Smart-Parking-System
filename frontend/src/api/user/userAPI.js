import api from "@/lib/axiosInstance";

const userAPI = {
  getUser: () => {
    return api.get("user/getInfo");
  }
};
export default userAPI;
