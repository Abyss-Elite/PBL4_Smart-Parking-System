"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import UserForm from "@/components/account_management/UserForm";
import userAPI from "@/api/user/userAPI";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import PATH from "@/routes/PATH";

export default function UserManagementPage() {
  const [users, setUsers] = useState([]);
  const [editingUser, setEditingUser] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  const initialUsers = [
    {
      id: 1,
      username: "user1",
      phoneNumber: "",
      email: "user1@gmail.com",
      role: { id: 1, name: "USER" },
    },
    {
      id: 2,
      username: "user2",
      phoneNumber: "",
      email: "user2@gmail.com",
      role: { id: 1, name: "USER" },
    },
    {
      id: 3,
      username: "user3",
      phoneNumber: "",
      email: "user3@gmail.com",
      role: { id: 1, name: "USER" },
    },
    {
      id: 4,
      username: "user4",
      phoneNumber: "",
      email: "user4@gmail.com",
      role: { id: 1, name: "USER" },
    },
    {
      id: 6,
      username: "admin123",
      phoneNumber: "",
      email: "admin123@gmail.com",
      role: { id: 3, name: "ADMIN" },
    },
    {
      id: 7,
      username: "user5",
      phoneNumber: "0905123456",
      email: "user5@gmail.com",
      role: { id: 1, name: "USER" },
    },
    {
      id: 9,
      username: "user6",
      phoneNumber: "0915123456",
      email: "user6@gmail.com",
      role: { id: 1, name: "USER" },
    },
  ];

  const fetchUsers = async () => {
    try {
      // const res = await userAPI.getUsers();
      // setUsers(res.data);
      setUsers(initialUsers);
    } catch (error) {
      console.error("Lỗi khi lấy user:", error);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const handleDelete = async (id) => {
    if (!confirm("Bạn có chắc muốn xóa user này?")) return;
    try {
      await userAPI.deleteUser(id);
      setUsers(users.filter((u) => u.id !== id));
    } catch (error) {
      console.error("Xóa user thất bại:", error);
    }
  };

  const handleSave = async (user) => {
    try {
      if (editingUser) {
        await userAPI.updateUser(user);
        setUsers(users.map((u) => (u.id === user.id ? user : u)));
      } else {
        const res = await userAPI.createUser(user);
        setUsers([...users, res.data]);
      }
      setEditingUser(null);
      setDialogOpen(false);
    } catch (error) {
      console.error("Lưu user thất bại:", error);
    }
  };

  const handleAddUser = () => {
    setEditingUser(null);
    setDialogOpen(true);
  };

  const handleEditUser = (user) => {
    setEditingUser(user);
    setDialogOpen(true);
  };

  return (
    <div className="container mx-auto px-4 py-6">
      <Card className="overflow-x-auto rounded-lg border border-gray-200 shadow-lg">
        <CardHeader className="mb-4 flex items-center justify-between">
          <CardTitle className="text-lg font-semibold">Quản lý tài khoản</CardTitle>
          <Button onClick={handleAddUser} className="bg-green-600 text-white hover:bg-green-700">
            Thêm user
          </Button>
        </CardHeader>
        <CardContent className="p-4">
          <table className="w-full min-w-[700px] overflow-hidden rounded-lg border border-gray-300 text-sm">
            <thead className="bg-gray-100 text-xs text-gray-700 uppercase">
              <tr>
                <th className="border-b border-gray-300 px-6 py-3">ID</th>
                <th className="border-b border-gray-300 px-6 py-3">Username</th>
                <th className="border-b border-gray-300 px-6 py-3">Email</th>
                <th className="border-b border-gray-300 px-6 py-3">Phone</th>
                <th className="border-b border-gray-300 px-6 py-3">Role</th>
                <th className="border-b border-gray-300 px-6 py-3">Chi tiết xe</th>
                <th className="border-b border-gray-300 px-6 py-3">Hành động</th>
              </tr>
            </thead>
            <tbody>
              {users.map((user) => (
                <tr
                  key={user.id}
                  className="border-b border-gray-200 transition-colors hover:bg-gray-50"
                >
                  <td className="px-6 py-3">{user.id}</td>
                  <td className="px-6 py-3">{user.username}</td>
                  <td className="px-6 py-3">{user.email}</td>
                  <td className="px-6 py-3">{user.phoneNumber || "-"}</td>
                  <td className="px-6 py-3">{user.role.name}</td>
                  <td className="px-6 py-3">
                    <Link
                      href={PATH.DASHBOARD.ACCOUNT_MANAGEMENT.VEHICLE(user.id)}
                      className="text-blue-600 hover:underline"
                    >
                      Xem xe
                    </Link>
                  </td>
                  <td className="flex space-x-2 px-6 py-3">
                    <Button size="sm" variant="outline" onClick={() => handleEditUser(user)}>
                      Edit
                    </Button>
                    <Button size="sm" variant="destructive" onClick={() => handleDelete(user.id)}>
                      Delete
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{editingUser ? "Chỉnh sửa user" : "Thêm user mới"}</DialogTitle>
          </DialogHeader>
          <UserForm user={editingUser} onSave={handleSave} />
        </DialogContent>
      </Dialog>
    </div>
  );
}
