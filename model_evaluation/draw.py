import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import seaborn as sns

# Set Seaborn theme for better aesthetics
sns.set_theme(context='paper', style='whitegrid', palette='deep', 
              font='Arial', font_scale=1.8, color_codes=True, 
              rc={'lines.linewidth': 2, 'axes.grid': True,
                  'ytick.left': True, 'xtick.bottom': True, 
                  'font.weight':'bold', 'axes.labelweight': 'bold'})

def plot(learner_name, label_name, y_train, predict_results_train, y_test, predict_results_test, sample_size_train=10000, sample_size_test=3000):
    """
    Plot both scatter plots and histograms for model performance.
    This version includes optimized 2D Density Plot and spine customization.
    """
    name = {'isp': 'I$_{sp}$(s)', 'c_t': 'T$_c$(K)', 'cstar': 'C$^*$(m s$^{-1}$)'}

    # Convert tensors to numpy arrays if necessary
    y_train = y_train.detach().cpu().numpy()
    predict_results_train = predict_results_train.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()
    predict_results_test = predict_results_test.detach().cpu().numpy()

    # Sample from the large datasets for faster plotting
    sample_indices_train = np.random.choice(len(y_train), size=sample_size_train, replace=False)
    y_train_sample = y_train[sample_indices_train]
    predict_results_train_sample = predict_results_train[sample_indices_train]

    sample_indices_test = np.random.choice(len(y_test), size=sample_size_test, replace=False)
    y_test_sample = y_test[sample_indices_test]
    predict_results_test_sample = predict_results_test[sample_indices_test]

    # Compute performance metrics
    r2_train = r2_score(y_train, predict_results_train)
    mae_train = mean_absolute_error(y_train, predict_results_train)
    rmse_train = sqrt(mean_squared_error(y_train, predict_results_train))

    r2_test = r2_score(y_test, predict_results_test)
    mae_test = mean_absolute_error(y_test, predict_results_test)
    rmse_test = sqrt(mean_squared_error(y_test, predict_results_test))

    # Create subplots: 1 row, 3 columns
    plt.figure(figsize=(24, 6))  # Adjusted size for a more compact layout

    # First subplot: Training data
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title('Training set', fontsize=20, weight='bold')
    hb_train = ax1.hexbin(y_train_sample, predict_results_train_sample, gridsize=70, cmap='Blues', mincnt=1,vmax=50)

    # Ideal line for both train and test data
    combined_min = min(np.min(y_train_sample), np.min(y_test_sample))  # Minimum of both training and test sets
    combined_max = max(np.max(y_train_sample), np.max(y_test_sample))
    ax1.plot(np.linspace(combined_min, combined_max, 100),
             np.linspace(combined_min, combined_max, 100), 'r--', linewidth=2.0)

    ax1.set_xlabel('Observation ' + name[label_name], fontsize=18, weight='bold')
    ax1.set_ylabel('Prediction ' + name[label_name], fontsize=18, weight='bold')
    ax1.legend(frameon=False, fontsize=18)

    # Annotate performance metrics on the training plot
    ax1.text(0.05, 0.95, f"R²: {r2_train:.3f}", transform=ax1.transAxes, fontsize=18, verticalalignment='top', horizontalalignment='left', color='black')
    ax1.text(0.05, 0.87, f"MAE: {mae_train:.3f}", transform=ax1.transAxes, fontsize=18, verticalalignment='top', horizontalalignment='left', color='black')
    ax1.text(0.05, 0.79, f"RMSE: {rmse_train:.3f}", transform=ax1.transAxes, fontsize=18, verticalalignment='top', horizontalalignment='left', color='black')

    # Apply spine style
    for spine in ['top', 'right', 'left', 'bottom']:
        ax1.spines[spine].set_color('k')
        ax1.spines[spine].set_linewidth(2)

    # Second subplot: Test data
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title('Test set', fontsize=20, weight='bold')
    hb_test = ax2.hexbin(y_test_sample, predict_results_test_sample, gridsize=70, cmap='Purples', mincnt=1, vmax=5)
    ax2.plot(np.linspace(combined_min, combined_max, 100),
             np.linspace(combined_min, combined_max, 100), 'r--', linewidth=2.0)

    ax2.set_xlabel('Observation ' + name[label_name], fontsize=18, weight='bold')
    ax2.set_ylabel('Prediction ' + name[label_name], fontsize=18, weight='bold')
    ax2.legend(frameon=False, fontsize=15)

    # Annotate performance metrics on the test plot
    ax2.text(0.05, 0.95, f"R²: {r2_test:.3f}", transform=ax2.transAxes, fontsize=18, verticalalignment='top', horizontalalignment='left', color='black')
    ax2.text(0.05, 0.87, f"MAE: {mae_test:.3f}", transform=ax2.transAxes, fontsize=18, verticalalignment='top', horizontalalignment='left', color='black')
    ax2.text(0.05, 0.79, f"RMSE: {rmse_test:.3f}", transform=ax2.transAxes, fontsize=18, verticalalignment='top', horizontalalignment='left', color='black')

    # Apply spine style
    for spine in ['top', 'right', 'left', 'bottom']:
        ax2.spines[spine].set_color('k')
        ax2.spines[spine].set_linewidth(2)

    # Third subplot: Deviation distribution (histogram)
    ax3 = plt.subplot(1, 3, 3)
    deviation_test = y_test - predict_results_test
    mu_test = np.mean(deviation_test)
    sigma_test = np.std(deviation_test)

    n_test, bins_test, patches_test = ax3.hist(deviation_test, bins=50, density=False, edgecolor='w', alpha=0.6, color='purple')
    y_test_distribution = (1 / (np.sqrt(2 * np.pi) * sigma_test)) * np.exp(-0.5 * (1 / sigma_test * (bins_test - mu_test))**2)
    y_test_scaled = np.max(n_test) * y_test_distribution / np.max(y_test_distribution)

    min_abs_value = min(np.abs(bins_test[0]), np.abs(bins_test[-1]))
    ax3.set_xlim(-min_abs_value, min_abs_value)  # Set x-axis symmetric range

    ax3.plot(bins_test, y_test_scaled, '--', linewidth=2, color='red')
    ax3.set_xlabel('Deviation ' + name[label_name], fontsize=18, weight='bold')
    ax3.set_ylabel('Number', fontsize=18, weight='bold')
    ax3.legend(frameon=False, fontsize=15)
    ax3.set_title('Test set', fontsize=20, weight='bold')
    # Apply spine style
    for spine in ['top', 'right', 'left', 'bottom']:
        ax3.spines[spine].set_color('k')
        ax3.spines[spine].set_linewidth(2)

    # Customize tick parameters for both axes and colorbars (bold and large font)
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='major', labelsize=15, width=2, direction='in', length=6, colors='black', grid_color='gray', grid_alpha=0.5)

    # Adjust layout for better spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust the spacing to make the layout more compact

    # Save and display the plot
    plt.savefig(f'{label_name}_{learner_name}.tif', bbox_inches='tight', dpi=600)
    plt.show()

def plot1(learner_name, label_name, y_train, predict_results_train, y_test, predict_results_test):
    """
    Plot both scatter plots and histograms for model performance.
    """
    # Convert tensors to numpy arrays if necessary
    y_train = y_train.detach().cpu().numpy()
    predict_results_train = predict_results_train.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()
    predict_results_test = predict_results_test.detach().cpu().numpy()
    
    font = {'family': 'Arial', 'weight': 'bold', 'size': 20}
    plt.figure(figsize=(16, 7))

    # Scatter plot (Training and Test data)
    plt.subplot(1, 2, 1)
    plt.grid(linestyle='--', linewidth=2, color='gray', alpha=0.4)
    plt.scatter(y_train, predict_results_train, c='deepskyblue', edgecolors='black', linewidth=0.1, label='Training')
    plt.scatter(y_test, predict_results_test, c='plum', edgecolors='black', linewidth=0.3, alpha=0.4, label='Test')

    # Plot the ideal line
    x = np.linspace(min(predict_results_train), max(predict_results_train), 5)
    plt.plot(x, x, 'r--', linewidth=2.0)
    plt.xlabel('Observation', fontdict=font)
    plt.ylabel('Prediction', fontdict=font)
    plt.legend(frameon=False, prop=font)
    
    # Histogram (Deviation between Prediction and Target)
    plt.subplot(1, 2, 2)
    deviation_test = y_test - predict_results_test
    mu_test = np.mean(deviation_test)
    sigma_test = np.std(deviation_test)
    
    n_test, bins_test, patches_test = plt.hist(deviation_test, bins=50, density=False, edgecolor='w', alpha=0.8, color='purple')
    y_test_distribution = (1 / (np.sqrt(2 * np.pi) * sigma_test)) * np.exp(-0.5 * (1 / sigma_test * (bins_test - mu_test))**2)
    y_test = np.max(n_test) * y_test_distribution / np.max(y_test_distribution)
    plt.plot(bins_test, y_test, '--', linewidth=2, color='red', label='Test')
    
    plt.xlabel('Deviation', fontdict=font)
    plt.ylabel('Number', fontdict=font)
    plt.legend(frameon=False, prop=font)
    
    # Customize plot appearance
    plt.tight_layout()
    plt.savefig(f'{label_name}_{learner_name}.tif', bbox_inches='tight', dpi=600)
    plt.show()

