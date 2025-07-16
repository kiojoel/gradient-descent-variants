import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.loss_functions import Rosenbrock, Quadratic
from src.optimizers import SGD, RMSprop, Adam
from src.utils.experiment_runner import run_optimization
import pandas as pd


# Function to plot loss

def plot_convergence(loss_fn, history):
    losses = [loss_fn.evaluate(p) for p in history]
    fig, ax = plt.subplots()
    ax.plot(losses, label="Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Convergence Plot")
    ax.legend()
    return fig, losses

# Function to plot 2D path

def plot_2d_path(loss_fn, history, xlim, ylim):
    x_vals = np.linspace(*xlim, 100)
    y_vals = np.linspace(*ylim, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([[loss_fn.evaluate(np.array([x, y])) for x in x_vals] for y in y_vals])

    fig, ax = plt.subplots()
    cp = ax.contour(X, Y, Z, levels=50, cmap='viridis')
    path = np.array(history)
    ax.plot(path[:, 0], path[:, 1], color='red', marker='o', markersize=2, label='Path')
    ax.set_title("2D Contour Path")
    ax.legend()
    return fig

# Function to plot 3D surface

def plot_3d_surface(loss_fn, history, xlim, ylim):
    x_vals = np.linspace(*xlim, 100)
    y_vals = np.linspace(*ylim, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([[loss_fn.evaluate(np.array([x, y])) for x in x_vals] for y in y_vals])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    path = np.array(history)
    ax.plot(path[:, 0], path[:, 1], [loss_fn.evaluate(p) for p in path], color='red', marker='o')
    ax.set_title("3D Optimization Path")
    return fig

st.set_page_config(
    page_title="Gradient Descent Visualizer",
    page_icon="⚡",
    layout="wide"
)
st.title("Gradient Descent Optimizer Visualizer")
st.markdown("An interactive tool to explore and compare optimization algorithms.")

# Sidebar configuration
loss_fn_choice = st.sidebar.selectbox("Choose Loss Function", ["Quadratic", "Rosenbrock"])
selected_optimizers = st.sidebar.multiselect("Select Optimizers to Compare", ["SGD", "RMSprop", "Adam"], default=["SGD", "Adam"])
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.0001, max_value=0.9, value=0.01, step=0.001, format="%.4f")
iterations = st.sidebar.slider("Iterations", min_value=10, max_value=2000, value=300, step=10)

# Define loss function
if loss_fn_choice == "Quadratic":
    loss_fn = Quadratic()
    xlim = (-6, 6)
    ylim = (-6, 6)
else:
    loss_fn = Rosenbrock()
    xlim = (-2, 2)
    ylim = (-1, 3)

optimizer_classes = {"SGD": SGD, "RMSprop": RMSprop, "Adam": Adam}

# Run optimization on button click
if st.button("Run Optimization"):
    start_point = np.array([-1.5, -1.0])
    results = {}
    export_data = []

    for opt_name in selected_optimizers:
        optimizer = optimizer_classes[opt_name](lr=learning_rate)
        history = run_optimization(loss_fn, optimizer, start_point, iterations)
        final_loss = loss_fn.evaluate(history[-1])
        results[opt_name] = {
            "history": history,
            "final_loss": final_loss,
            "final_params": history[-1],
        }
        export_data.append({
            "Optimizer": opt_name,
            "Final Loss": final_loss,
            "Final x": history[-1][0],
            "Final y": history[-1][1],
        })

    st.write("### Final Results Summary")
    st.dataframe(pd.DataFrame(export_data))
    csv = pd.DataFrame(export_data).to_csv(index=False).encode('utf-8')
    st.download_button("Download Results as CSV", data=csv, file_name="optimizer_results.csv", mime="text/csv")

    for opt_name, result in results.items():
        st.subheader(f"Results for {opt_name}")

        col1, col2 = st.columns(2)
        with col1:
            fig_convergence, losses = plot_convergence(loss_fn, result["history"])
            st.pyplot(fig_convergence)
            st.line_chart(losses, height=200, use_container_width=True)

        with col2:
            st.pyplot(plot_2d_path(loss_fn, result["history"], xlim, ylim))

        st.pyplot(plot_3d_surface(loss_fn, result["history"], xlim, ylim))

        st.write("**Final Parameters:**", result["final_params"])
        st.write("**Final Loss:**", f"{result['final_loss']:.8f}")
