"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { userAPI } from "@/api/user/userAPI";
import { Edit2, Trash2, X } from "lucide-react";

export default function AllUsersTable() {
  const [users, setUsers] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedUser, setSelectedUser] = useState(null);
  const [formData, setFormData] = useState({});
  const [notification, setNotification] = useState(null);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [userToDelete, setUserToDelete] = useState(null);
  const router = useRouter();

  const fetchUsers = async () => {
    try {
      const res = await userAPI.getUsers();
      setUsers(res.data);
    } catch (err) {
      console.error(err);
      setNotification({ type: "error", message: "Lỗi khi lấy danh sách user" });
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const handleEdit = (user) => {
    setSelectedUser(user);
    setFormData({
      username: user.username || "",
      email: user.email || "",
      phoneNumber: user.phoneNumber || "",
      roleId: user.role?.id || 2,
      status: user.status || "ACTIVE",
    });
    setModalOpen(true);
  };

  const handleUpdate = async () => {
    try {
      await userAPI.updateUser({ id: selectedUser.id, ...formData });
      setNotification({ type: "success", message: "Cập nhật thành công" });
      setModalOpen(false);
      fetchUsers();
    } catch (err) {
      console.error(err);
      setNotification({ type: "error", message: "Cập nhật thất bại" });
    }
  };

  const openDeleteModal = (user) => {
    setUserToDelete(user);
    setDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    try {
      await userAPI.deleteUser(userToDelete.id);
      setNotification({ type: "success", message: `Đã xóa user ${userToDelete.username}` });
      setDeleteModalOpen(false);
      fetchUsers();
    } catch (err) {
      console.error(err);
      setNotification({ type: "error", message: "Xóa user thất bại" });
    }
  };

  return (
    <div className="overflow-x-auto p-4">
      {notification && (
        <div
          className={`mb-4 rounded p-3 text-white ${
            notification.type === "success" ? "bg-green-400" : "bg-red-400"
          }`}
        >
          {notification.message}
        </div>
      )}

      <table className="min-w-full divide-y divide-gray-200 rounded-xl bg-white text-base shadow-lg">
        <thead className="bg-gray-100">
          <tr>
            <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700">ID</th>
            <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700">Username</th>
            <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700">Email</th>
            <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700">Phone</th>
            <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700">Role</th>
            <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700">Status</th>
            <th className="px-4 py-2 text-center text-sm font-semibold text-gray-700">Actions</th>
          </tr>
        </thead>

        <tbody className="divide-y divide-gray-200">
          {users.map((user) => (
            <tr
              key={user.id}
              className="cursor-pointer transition hover:bg-green-50"
              onClick={() => router.push(`/dashboard/employee/${user.id}`)}
            >
              <td className="px-4 py-2 text-gray-700">{user.id}</td>
              <td className="px-4 py-2 font-medium text-gray-700">{user.username}</td>
              <td className="px-4 py-2 text-gray-700">{user.email}</td>
              <td className="px-4 py-2 text-gray-700">{user.phoneNumber}</td>
              <td className="px-4 py-2 text-gray-700">{user.role?.name}</td>
              <td className="px-4 py-2 text-gray-700">{user.status}</td>
              <td className="flex justify-center gap-2 px-4 py-2">
                <button
                  onClick={(e) => {
                    e.stopPropagation(); // Ngăn event từ tr.row ảnh hưởng
                    handleEdit(user);
                  }}
                  className="flex cursor-pointer items-center gap-1 rounded bg-green-500 px-3 py-1 text-white transition hover:bg-green-600"
                >
                  <Edit2 size={16} />
                  Edit
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    openDeleteModal(user);
                  }}
                  className="flex cursor-pointer items-center gap-1 rounded bg-red-500 px-3 py-1 text-white transition hover:bg-red-600"
                >
                  <Trash2 size={16} />
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {deleteModalOpen && (
        <div className="bg-opacity-40 fixed inset-0 z-50 flex items-center justify-center bg-gray-500/45">
          <div className="relative w-full max-w-md rounded-xl bg-white p-6 shadow-lg">
            <div className="mb-4">
              <h2 className="text-xl font-bold text-red-600">Xoá người dùng</h2>
              <p className="mt-2 text-gray-700">
                Bạn có chắc chắn muốn xoá{" "}
                <span className="font-semibold">{userToDelete?.username}</span> không?
              </p>
            </div>

            <div className="mt-6 flex justify-end gap-3">
              <button
                onClick={() => setDeleteModalOpen(false)}
                className="cursor-pointer rounded bg-gray-300 px-4 py-2 transition hover:bg-gray-400"
              >
                Hủy
              </button>

              <button
                onClick={confirmDelete}
                className="cursor-pointer rounded bg-red-500 px-4 py-2 text-white transition hover:bg-red-600"
              >
                Xoá
              </button>
            </div>
          </div>
        </div>
      )}

      {modalOpen && (
        <div className="bg-opacity-40 fixed inset-0 z-50 flex items-center justify-center bg-gray-500/45">
          <div className="relative w-full max-w-md rounded-xl bg-white p-6 shadow-lg">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-bold">Chỉnh sửa User</h2>
              <button onClick={() => setModalOpen(false)}>
                <X size={20} className="text-gray-500 hover:text-gray-700" />
              </button>
            </div>

            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-gray-700">Username</label>
                <input
                  type="text"
                  className="w-full rounded border p-2"
                  value={formData.username}
                  onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                />
              </div>

              <div>
                <label className="mb-1 block text-gray-700">Email</label>
                <input
                  type="email"
                  className="w-full rounded border p-2"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                />
              </div>

              <div>
                <label className="mb-1 block text-gray-700">Phone</label>
                <input
                  type="text"
                  className="w-full rounded border p-2"
                  value={formData.phoneNumber}
                  onChange={(e) => setFormData({ ...formData, phoneNumber: e.target.value })}
                />
              </div>

              <div>
                <label className="mb-1 block text-gray-700">Role</label>
                <select
                  className="w-full rounded border p-2"
                  value={formData.roleId}
                  onChange={(e) => setFormData({ ...formData, roleId: Number(e.target.value) })}
                >
                  <option value={1}>ADMIN</option>
                  <option value={2}>STAFF</option>
                </select>
              </div>

              <div>
                <label className="mb-1 block text-gray-700">Status</label>
                <select
                  className="w-full rounded border p-2"
                  value={formData.status}
                  onChange={(e) => setFormData({ ...formData, status: e.target.value })}
                >
                  <option value="ACTIVE">ACTIVE</option>
                  <option value="INACTIVE">INACTIVE</option>
                </select>
              </div>
            </div>

            <div className="mt-6 flex justify-end gap-3">
              <button
                onClick={() => setModalOpen(false)}
                className="cursor-pointer rounded bg-gray-300 px-4 py-2 transition hover:bg-gray-400"
              >
                Hủy
              </button>
              <button
                onClick={handleUpdate}
                className="cursor-pointer rounded bg-green-400 px-4 py-2 text-white transition hover:bg-green-600"
              >
                Lưu
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
