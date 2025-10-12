package com.example.backend.validator;

import com.example.backend.repository.UserRepository;
import com.example.backend.model.User;
import org.springframework.stereotype.Component;

@Component
public class UserValidator {

    private final UserRepository userRepository;

    public UserValidator(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public void validateUsername(String username, Long excludeId) {
        if (username == null || username.isBlank()) {
            throw new RuntimeException("Tên đăng nhập không được để trống");
        }
        userRepository.findByUsername(username)
            .filter(u -> !u.getId().equals(excludeId))
            .ifPresent(u -> { throw new RuntimeException("Tên đăng nhập đã tồn tại."); });
    }

    public void validateEmail(String email, Long excludeId) {
        if (email == null || email.isBlank()) {
            throw new RuntimeException("Email không được để trống");
        }
        if (!email.matches("^[A-Za-z0-9+_.-]+@(.+)$")) {
            throw new RuntimeException("Email không hợp lệ.");
        }
        userRepository.findByEmail(email)
            .filter(u -> !u.getId().equals(excludeId))
            .ifPresent(u -> { throw new RuntimeException("Email đã tồn tại."); });
    }

    public void validatePhoneNumber(String phoneNumber, Long excludeId) {
        if (phoneNumber == null || phoneNumber.isBlank()) {
            throw new RuntimeException("Số điện thoại không được để trống");
        }
        userRepository.findByPhoneNumber(phoneNumber)
            .filter(u -> !u.getId().equals(excludeId))
            .ifPresent(u -> { throw new RuntimeException("Số điện thoại đã được đăng ký."); });
    }

    public void validatePassword(String password) {
        if (password == null || password.isBlank()) {
            throw new RuntimeException("Mật khẩu không được để trống");
        }
        if (password.length() < 8) {
            throw new RuntimeException("Mật khẩu phải ít nhất 8 ký tự.");
        }
        if (!password.matches(".*[!@#$%^&*(),.?\":{}|<>].*")) {
            throw new RuntimeException("Mật khẩu phải có ít nhất 1 ký tự đặc biệt.");
        }
    }
}
