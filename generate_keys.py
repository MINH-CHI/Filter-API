import secrets
import string

def generate_api_key(prefix="sk", length=32):
    alphabet = string.ascii_letters + string.digits
    random_string = ''.join(secrets.choice(alphabet) for _ in range(length))
    return f"{prefix}_{random_string}"

# Danh sách người dùng
users = [
    ("Sếp khánh", "Data_team"),
    ("Anh Khôi", "Data_team"),
    ("Vương", "AI_team"),
    ("Mạnh", "AI_team"),
    ("Minh","Data_team")
]

# Nội dung file config sẽ được tạo ra
file_content = "API_KEYS = {\n"
for name, prefix in users:
    key = generate_api_key(prefix=prefix)
    file_content += f'    "{key}": "{name}",\n'
file_content += "}\n"

with open("secrets_config.py", "w", encoding="utf-8") as f:
    f.write(file_content)

print("✅ Đã tạo xong file 'secrets_config.py' chứa Key!")