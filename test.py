import streamlit as st
import time

# Set the session timeout (in seconds)
session_timeout = 30  # 5 minutes

print("Hello World")

# Define a function to reset the session state
def reset_session_state():
    st.session_state.last_activity_time = time.time()


# Initialize session state variables
if "last_activity_time" not in st.session_state:
    st.session_state.last_activity_time = time.time()

# Check for inactivity and reset the session state if needed
if time.time() - st.session_state.last_activity_time > session_timeout:
    st.write("Session timeout due to inactivity.")
    st.reset_session_state()

# Display your app content
st.title("Session Timeout Example")
st.write("This is your Streamlit app content.")


# Update the last activity time whenever there's user interaction
# For example, when the user clicks a button or interacts with widgets
if st.button("Interact"):
    reset_session_state()
