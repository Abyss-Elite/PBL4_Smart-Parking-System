"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import authAPI from "@/api/auth/authApi";
import PATH from "@/routes/PATH";

export default function ResetPasswordForm({ onBack }) {
  const router = useRouter();
  const [resetForm, setResetForm] = useState({
    username: "",
    email: "",
    newPassword: "",
    confirmPassword: "",
  });
  const [resetError, setResetError] = useState("");

  const handleResetPassword = async () => {
    const { username, email, newPassword, confirmPassword } = resetForm;

    if (newPassword !== confirmPassword) {
      setResetError("Mật khẩu xác nhận không khớp.");
      return;
    }
    if (newPassword.length < 8) {
      setResetError("Mật khẩu mới phải có ít nhất 8 ký tự.");
      return;
    }
    if (!/[!@#$%^&*(),.?\":{}|<>]/.test(newPassword)) {
      setResetError("Mật khẩu phải chứa ít nhất 1 ký tự đặc biệt.");
      return;
    }

    try {
      await authAPI.resetPassword({ username, email, newPassword });
      setResetError("");
      router.push(PATH.LOGIN);
    } catch (err) {
      setResetError(err.response?.data?.message || "Không thể đặt lại mật khẩu.");
    }
  };

  return (
    <Card className="w-full max-w-sm border-none shadow-lg">
      <CardHeader>
        <CardTitle className="text-center text-2xl font-bold text-gray-800">
          Đặt lại mật khẩu
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          placeholder="Tên đăng nhập"
          value={resetForm.username}
          onChange={(e) => setResetForm({ ...resetForm, username: e.target.value })}
        />
        <Input
          placeholder="Email"
          type="email"
          value={resetForm.email}
          onChange={(e) => setResetForm({ ...resetForm, email: e.target.value })}
        />
        <Input
          placeholder="Mật khẩu mới"
          type="password"
          value={resetForm.newPassword}
          onChange={(e) => setResetForm({ ...resetForm, newPassword: e.target.value })}
        />
        <Input
          placeholder="Xác nhận mật khẩu"
          type="password"
          value={resetForm.confirmPassword}
          onChange={(e) => setResetForm({ ...resetForm, confirmPassword: e.target.value })}
        />
        {resetError && (
          <Alert variant="destructive">
            <AlertDescription>{resetError}</AlertDescription>
          </Alert>
        )}
        <div className="flex justify-between">
          <Button
            onClick={handleResetPassword}
            className="w-1/2 bg-blue-600 font-bold text-white hover:bg-blue-700"
          >
            Xác nhận
          </Button>
          <Button variant="outline" className="ml-2 w-1/2" onClick={onBack}>
            Trở lại
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
