import streamlit as st
import sqlite3

# ✅ Initialize database
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

# ✅ Create users table with full name
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        phone TEXT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
''')
conn.commit()

# Function to register user
def register_user(full_name, age, gender, phone, email, password):
    try:
        c.execute("INSERT INTO users (full_name, age, gender, phone, email, password) VALUES (?, ?, ?, ?, ?, ?)", 
                  (full_name, age, gender, phone, email, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Email already exists

# ✅ Fix: Function to verify login & retrieve full name
def login_user(email, password):
    c.execute("SELECT full_name FROM users WHERE email=? AND password=?", (email, password))
    user = c.fetchone()
    return user[0] if user else None  # ✅ Return full name, not None

# 🔒 Function to handle login and registration
def user_authentication():
    # ✅ Ensure session state is initialized
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["user_name"] = None

    # If already logged in, return True
    if st.session_state["logged_in"]:
        return True

    st.title("🔒 Welcome to Smart Disease Diagnosis")

    option = st.radio("Select an option:", ["Login", "Register"], key="auth_radio")

    if option == "Register":
        st.subheader("📝 Create a New Account")

        # ✅ Collect user details
        full_name = st.text_input("👤 Full Name", key="full_name_input")
        age = st.number_input("🎂 Age", min_value=1, max_value=120, step=1, key="age_input")
        gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"], key="gender_input")
        phone = st.text_input("📞 Phone Number", key="phone_input")
        email = st.text_input("📧 Gmail Address", key="email_input")
        password = st.text_input("🔑 Password", type="password", key="password_input")

        if st.button("Register", key="register_button"):
            if not email.endswith("@gmail.com"):
                st.error("❌ Please enter a valid Gmail address.")
            elif register_user(full_name, age, gender, phone, email, password):
                st.success("✅ Registration successful! Please log in.")
            else:
                st.error("⚠️ This email is already registered. Try logging in.")

    elif option == "Login":
        st.subheader("🔑 Login to Your Account")
        email = st.text_input("📧 Gmail Address", key="login_email_input")
        password = st.text_input("🔑 Password", type="password", key="login_password_input")

        if st.button("Login", key="login_button"):
            full_name = login_user(email, password)
            if full_name:
                st.session_state["logged_in"] = True
                st.session_state["user_name"] = full_name  # ✅ Fix: Store full name instead of None
                st.success(f"✅ Welcome, {full_name}!")
                st.rerun()  # Refresh page to load the app
            else:
                st.error("❌ Incorrect email or password.")

    return False  # User not logged in
