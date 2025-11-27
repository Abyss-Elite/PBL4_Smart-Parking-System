import api from "@/lib/axiosInstance";

export const staffManagementAPI = {
  getStaffList: () => {
    return api.get("/shifts/list");
  },
};
