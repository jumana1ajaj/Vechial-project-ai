#import the libraries
import math
import random
import time
import tkinter as tk
from tkinter import ttk, messagebox
from copy import deepcopy

#define the package class and the attributes
class Package:
    def __init__(self, id, x, y, weight, priority):
        self.id = id
        self.x = x
        self.y = y
        self.weight = weight
        self.priority = priority

    def __repr__(self):
        return f"P{self.id}({self.weight}kg, prio:{self.priority})"

#define the vehicle class
class Vehicle:
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity
        self.packages = []
        self.route = []

    def __repr__(self):
        return f"V{self.id}({self.capacity}kg)"


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_route_distance(route):
    if not route:
        return 0

    distance = 0
    prev_x, prev_y = 0, 0  # Start at inital

    for package in route:
        distance += euclidean_distance(prev_x, prev_y, package.x, package.y)
        prev_x, prev_y = package.x, package.y

    # Return to depot
    distance += euclidean_distance(prev_x, prev_y, 0, 0)
    return distance


def calculate_total_distance(solution):
    total_distance = 0
    for vehicle in solution:
        total_distance += calculate_route_distance(vehicle.route)
    return total_distance


def generate_initial_solution(packages, vehicles):
    """Generate initial solution with proper priority handling"""
    if not vehicles or not packages:#f theres no vehcials and packeges then return null
        return []

    # Sort packages by priority then by distance
    sorted_packages = sorted(packages,
                             key=lambda p: (p.priority, euclidean_distance(0, 0, p.x, p.y)))

    # Create vehicle copies
    vehicle_copies = [Vehicle(v.id, v.capacity) for v in vehicles]

    # Assign packages using priority-aware first-fit
    for package in sorted_packages:
        assigned = False # thats mean this bakedge still need vechial

        # Try vehicles in order of most available capacity first and lambda its instead of def
        for vehicle in sorted(vehicle_copies,
                              key=lambda v: v.capacity - sum(p.weight for p in v.packages),
                              reverse=True):
            current_load = sum(p.weight for p in vehicle.packages)
            if current_load + package.weight <= vehicle.capacity:
                vehicle.packages.append(package)
                assigned = True
                break

        if not assigned:
            messagebox.showwarning("Warning",
                                   f"Could not assign package {package.id} (weight: {package.weight}kg, priority: {package.priority})")
            continue

    # Calculate routes using nearest neighbor
    for vehicle in vehicle_copies:
        vehicle.route = nearest_neighbor_route(vehicle.packages)

    return vehicle_copies


def nearest_neighbor_route(packages):
    if not packages:
        return []

    unvisited = packages.copy()
    route = [] # save last version of the packeges
    current_x, current_y = 0, 0  # Start at inital

    while unvisited:
        # Find the unvisted packeges
        nearest = min(unvisited, key=lambda p: euclidean_distance(current_x, current_y, p.x, p.y))
        route.append(nearest)
        current_x, current_y = nearest.x, nearest.y
        unvisited.remove(nearest)

    return route


def generate_neighbor_solution(current_solution):
    neighbor = deepcopy(current_solution)
    operation = random.choice(['swap', 'move', 'reverse_route', 'reassign'])

    if operation == 'swap' and len(neighbor) > 1:
        v1, v2 = random.sample(neighbor, 2)
        if v1.packages and v2.packages:
            idx1 = random.randint(0, len(v1.packages) - 1)
            idx2 = random.randint(0, len(v2.packages) - 1)

            # Check capacity constraints
            p1, p2 = v1.packages[idx1], v2.packages[idx2]
            if (sum(p.weight for p in v1.packages) - p1.weight + p2.weight <= v1.capacity and
                    sum(p.weight for p in v2.packages) - p2.weight + p1.weight <= v2.capacity):
                v1.packages[idx1], v2.packages[idx2] = v2.packages[idx2], v1.packages[idx1]

    elif operation == 'move' and len(neighbor) > 1:
        v1, v2 = random.sample(neighbor, 2)
        if v1.packages:
            p = random.choice(v1.packages)
            if sum(pkg.weight for pkg in v2.packages) + p.weight <= v2.capacity:
                v1.packages.remove(p)
                v2.packages.append(p)

    elif operation == 'reverse_route':
        # Reverse a segment of a vehicle's route
        v = random.choice(neighbor)
        if len(v.route) >= 2:
            start = random.randint(0, len(v.route) - 2)
            end = random.randint(start + 1, len(v.route) - 1)
            v.route[start:end + 1] = reversed(v.route[start:end + 1])

    elif operation == 'reassign':
        # Reassign a random package to a random vehicle
        all_packages = [p for v in neighbor for p in v.packages]
        if all_packages:
            p = random.choice(all_packages)
            current_vehicle = next(v for v in neighbor if p in v.packages)
            new_vehicle = random.choice(neighbor)

            if (new_vehicle != current_vehicle and
                    sum(pkg.weight for pkg in new_vehicle.packages) + p.weight <= new_vehicle.capacity):
                current_vehicle.packages.remove(p)
                new_vehicle.packages.append(p)
    for vehicle in neighbor:
        if operation in ['swap', 'move', 'reassign']:
            if vehicle.packages:
                vehicle.route = nearest_neighbor_route(vehicle.packages)
            else:
                vehicle.route = []

    return neighbor


def simulated_annealing(packages, vehicles, initial_temp=1000, cooling_rate=0.95,
                        stopping_temp=1, iterations_per_temp=100, progress_callback=None):
    current_solution = generate_initial_solution(packages, vehicles)
    if current_solution is None:
        return None

    current_cost = calculate_total_distance(current_solution)


    best_solution = deepcopy(current_solution)
    best_cost = current_cost

    temp = initial_temp
    iteration = 0
    total_iterations = math.ceil(math.log(stopping_temp / initial_temp, cooling_rate)) * iterations_per_temp

    while temp > stopping_temp:
        for _ in range(iterations_per_temp):
            iteration += 1
            if progress_callback:
                progress = min(100, int(iteration / total_iterations * 100))
                progress_callback(progress)

            neighbor = generate_neighbor_solution(current_solution)
            neighbor_cost = calculate_total_distance(neighbor)

            cost_diff = neighbor_cost - current_cost

            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temp):
                current_solution = deepcopy(neighbor)
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = deepcopy(current_solution)
                    best_cost = current_cost

        temp *= cooling_rate

    return best_solution

    def genetic_algorithm(packages, vehicles, population_size=100, mutation_rate=0.02, generations=200,
                          progress_callback=None):
     def create_valid_individual():
            """Create a valid individual using priority-based assignment"""
            individual = [Vehicle(v.id, v.capacity) for v in vehicles]
            sorted_packages = sorted(packages, key=lambda p: (p.priority, -p.weight))

            for package in sorted_packages:
                assigned = False

                for vehicle in sorted(individual,
                                      key=lambda v: v.capacity - sum(p.weight for p in v.packages),
                                      reverse=True):
                    if sum(p.weight for p in vehicle.packages) + package.weight <= vehicle.capacity:
                        vehicle.packages.append(package)
                        assigned = True
                        break

                if not assigned:
                    least_loaded = min(individual, key=lambda v: sum(p.weight for p in v.packages))
                    least_loaded.packages.append(package)
            for vehicle in individual:
                vehicle.route = nearest_neighbor_route(vehicle.packages)
            return individual

    def fitness(individual):
        total_distance = calculate_total_distance(individual)
        penalty = 0
        for vehicle in individual:
            if sum(p.weight for p in vehicle.packages) > vehicle.capacity:
                penalty += 1000
        return 1 / (1 + total_distance + penalty)

    def crossover(parent1, parent2):
        child = [Vehicle(v.id, v.capacity) for v in vehicles]
        for package in packages:
            src_parent = random.choice([parent1, parent2])
            src_vehicle = next(v for v in src_parent if package in v.packages)
            dest_vehicle = next(v for v in child if v.id == src_vehicle.id)
            if sum(p.weight for p in dest_vehicle.packages) + package.weight <= dest_vehicle.capacity:
                dest_vehicle.packages.append(package)
        unassigned = [p for p in packages if not any(p in v.packages for v in child)]
        for package in unassigned:
            best_vehicle = max(child,
                               key=lambda v: v.capacity - sum(p.weight for p in v.packages))
            best_vehicle.packages.append(package)
        for vehicle in child:
            vehicle.route = nearest_neighbor_route(vehicle.packages)
        return child

    def mutate(individual):
        new_individual = deepcopy(individual)
        operation = random.choice(['swap', 'move', 'reverse_route'])

        if operation == 'swap' and len(new_individual) > 1:
            v1, v2 = random.sample(new_individual, 2)
            if v1.packages and v2.packages:
                idx1 = random.randint(0, len(v1.packages) - 1)
                idx2 = random.randint(0, len(v2.packages) - 1)
                p1, p2 = v1.packages[idx1], v2.packages[idx2]
                if (sum(p.weight for p in v1.packages) - p1.weight + p2.weight <= v1.capacity and
                        sum(p.weight for p in v2.packages) - p2.weight + p1.weight <= v2.capacity):
                    v1.packages[idx1], v2.packages[idx2] = p2, p1

        elif operation == 'move' and len(new_individual) > 1:
            v1, v2 = random.sample(new_individual, 2)
            if v1.packages:
                p = random.choice(v1.packages)
                if sum(pkg.weight for pkg in v2.packages) + p.weight <= v2.capacity:
                    v1.packages.remove(p)
                    v2.packages.append(p)

        elif operation == 'reverse_route':
            v = random.choice(new_individual)
            if len(v.route) >= 2:
                start = random.randint(0, len(v.route) - 2)
                end = random.randint(start + 1, len(v.route) - 1)
                v.route[start:end + 1] = reversed(v.route[start:end + 1])

        # Recalculate routes if needed
        if operation in ['swap', 'move']:
            for vehicle in new_individual:
                vehicle.route = nearest_neighbor_route(vehicle.packages)

        return new_individual
    population = [create_valid_individual() for _ in range(population_size)]

    for generation in range(generations):
        if progress_callback:
            progress = min(100, int(generation / generations * 100))
            progress_callback(progress)
        fitnesses = [fitness(ind) for ind in population]
        new_population = []
        elites = sorted(zip(population, fitnesses), key=lambda x: -x[1])[:population_size // 10]
        new_population.extend([ind for ind, fit in elites])
        while len(new_population) < population_size:

            parents = []
            for _ in range(2):
                candidates = random.sample(list(zip(population, fitnesses)), k=3)
                best = max(candidates, key=lambda x: x[1])[0]
                parents.append(best)
            child = crossover(parents[0], parents[1])
            if random.random() < mutation_rate:
                child = mutate(child)

            new_population.append(child)

        population = new_population
    best_individual = max(population, key=lambda x: fitness(x))
    return best_individual

    def generate_empty_solution():
        return []



    def select_parents(population, fitnesses):
        parents = []
        for _ in range(2):  # Select 2 parents
            candidates = random.sample(list(zip(population, fitnesses)), k=3)
            winner = max(candidates, key=lambda x: x[1])[0]
            parents.append(winner)
        return parents
    population = [create_individual() for _ in range(population_size)]

    for generation in range(generations):
        if progress_callback:
            progress = min(100, int(generation / generations * 100))
            progress_callback(progress)
        fitnesses = [fitness(ind) for ind in population]
        new_population = []
        for _ in range(population_size // 2):
            parents = select_parents(population, fitnesses)
            offspring1 = crossover(parents[0], parents[1])
            offspring2 = crossover(parents[1], parents[0])
            new_population.extend([mutate(offspring1), mutate(offspring2)])
        population = sorted(population + new_population, key=lambda x: -fitness(x))[:population_size]

    return max(population, key=lambda x: fitness(x))


class DeliveryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Delivery Optimization System")
        self.root.geometry("1200x800")

        self.packages = []
        self.vehicles = []
        self.running = False
        self.current_solution = None

        self.create_widgets()

    def create_widgets(self):
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        left_panel = tk.Frame(main_pane)
        main_pane.add(left_panel)
        right_panel = tk.Frame(main_pane)
        main_pane.add(right_panel)
        self.create_input_panel(left_panel)
        self.create_algorithm_panel(left_panel)
        self.create_visualization_panel(right_panel)
        self.create_results_panel(right_panel)

    def create_input_panel(self, parent):
        vehicle_frame = tk.LabelFrame(parent, text="Vehicle Configuration", padx=5, pady=5)
        vehicle_frame.pack(fill=tk.X, padx=5, pady=5)
        self.vehicle_tree = ttk.Treeview(vehicle_frame, columns=('id', 'capacity'), show='headings', height=4)
        self.vehicle_tree.heading('id', text='Vehicle ID')
        self.vehicle_tree.heading('capacity', text='Capacity (kg)')
        self.vehicle_tree.column('id', width=100)
        self.vehicle_tree.column('capacity', width=100)
        self.vehicle_tree.pack(fill=tk.X)
        vehicle_controls = tk.Frame(vehicle_frame)
        vehicle_controls.pack(fill=tk.X)

        tk.Label(vehicle_controls, text="Capacity:").pack(side=tk.LEFT)
        self.vehicle_capacity_entry = tk.Entry(vehicle_controls, width=10)
        self.vehicle_capacity_entry.pack(side=tk.LEFT)
        self.vehicle_capacity_entry.insert(0, "100")

        tk.Button(vehicle_controls, text="Add Vehicle", command=self.add_vehicle).pack(side=tk.LEFT, padx=5)
        tk.Button(vehicle_controls, text="Remove Selected", command=self.remove_vehicle).pack(side=tk.LEFT)

        # Package Configuration Frame
        package_frame = tk.LabelFrame(parent, text="Package Configuration", padx=5, pady=5)
        package_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Package Treeview
        self.package_tree = ttk.Treeview(package_frame,
                                         columns=('id', 'x', 'y', 'weight', 'priority'),
                                         show='headings',
                                         height=8)
        self.package_tree.heading('id', text='ID')
        self.package_tree.heading('x', text='X Coord')
        self.package_tree.heading('y', text='Y Coord')
        self.package_tree.heading('weight', text='Weight (kg)')
        self.package_tree.heading('priority', text='Priority')

        for col in ('id', 'x', 'y', 'weight', 'priority'):
            self.package_tree.column(col, width=80, anchor=tk.CENTER)

        self.package_tree.pack(fill=tk.BOTH, expand=True)

        # Package Controls
        package_controls = tk.Frame(package_frame)
        package_controls.pack(fill=tk.X)

        # Package Entry Fields
        entry_frame = tk.Frame(package_controls)
        entry_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(entry_frame, text="X:").pack(side=tk.LEFT)
        self.pkg_x_entry = tk.Entry(entry_frame, width=8)
        self.pkg_x_entry.pack(side=tk.LEFT)

        tk.Label(entry_frame, text="Y:").pack(side=tk.LEFT)
        self.pkg_y_entry = tk.Entry(entry_frame, width=8)
        self.pkg_y_entry.pack(side=tk.LEFT)

        tk.Label(entry_frame, text="Weight:").pack(side=tk.LEFT)
        self.pkg_weight_entry = tk.Entry(entry_frame, width=8)
        self.pkg_weight_entry.pack(side=tk.LEFT)

        tk.Label(entry_frame, text="Priority:").pack(side=tk.LEFT)
        self.pkg_priority_var = tk.StringVar(value="3")
        priority_menu = tk.OptionMenu(entry_frame, self.pkg_priority_var, "1", "2", "3", "4", "5")
        priority_menu.pack(side=tk.LEFT)

        # Package Buttons Frame
        button_frame = tk.Frame(package_controls)
        button_frame.pack(side=tk.RIGHT)

        # Package Action Buttons
        tk.Button(button_frame, text="Add Package", command=self.add_package).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Remove Selected", command=self.remove_package).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Clear All", command=self.clear_packages).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Generate Random", command=self.generate_random_packages).pack(side=tk.LEFT,
                                                                                                    padx=5)

    def create_algorithm_panel(self, parent):
        algo_frame = tk.LabelFrame(parent, text="Algorithm Configuration", padx=5, pady=5)
        algo_frame.pack(fill=tk.X, padx=5, pady=5)

        # Algorithm Selection
        self.algorithm_var = tk.StringVar(value="simulated_annealing")

        tk.Radiobutton(algo_frame, text="Simulated Annealing", variable=self.algorithm_var,
                       value="simulated_annealing", command=self.update_algorithm_params).pack(anchor=tk.W)
        tk.Radiobutton(algo_frame, text="Genetic Algorithm", variable=self.algorithm_var,
                       value="genetic_algorithm", command=self.update_algorithm_params).pack(anchor=tk.W)

        # Algorithm Parameters Frame
        self.params_frame = tk.Frame(algo_frame)
        self.params_frame.pack(fill=tk.X, pady=5)

        # Simulated Annealing Parameters (default)
        self.sa_cooling_label = tk.Label(self.params_frame, text="Cooling Rate (0.90-0.99):")
        self.sa_cooling_label.pack(anchor=tk.W)

        self.sa_cooling_slider = tk.Scale(self.params_frame, from_=90, to=99, orient=tk.HORIZONTAL)
        self.sa_cooling_slider.set(95)
        self.sa_cooling_slider.pack(fill=tk.X)

        # Genetic Algorithm Parameters (hidden initially)
        self.ga_population_label = tk.Label(self.params_frame, text="Population Size (50-100):")
        self.ga_population_entry = tk.Entry(self.params_frame)
        self.ga_population_entry.insert(0, "50")

        self.ga_mutation_label = tk.Label(self.params_frame, text="Mutation Rate (0.01-0.1):")
        self.ga_mutation_entry = tk.Entry(self.params_frame)
        self.ga_mutation_entry.insert(0, "0.05")

        # Execution Controls
        exec_frame = tk.Frame(algo_frame)
        exec_frame.pack(fill=tk.X, pady=5)

        self.progress = ttk.Progressbar(exec_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, pady=2)

        button_frame = tk.Frame(exec_frame)
        button_frame.pack(fill=tk.X)

        self.run_button = tk.Button(button_frame, text="Run Optimization", command=self.run_optimization)
        self.run_button.pack(side=tk.LEFT, padx=2)

        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_optimization, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=2)

        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset_solution)
        self.reset_button.pack(side=tk.LEFT, padx=2)
        self.update_algorithm_params()

    def create_visualization_panel(self, parent):
        vis_frame = tk.LabelFrame(parent, text="Route Visualization", padx=5, pady=5)
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas = tk.Canvas(vis_frame, bg='white', width=500, height=500)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        for i in range(0, 101, 10):
            x = i * width / 100
            self.canvas.create_line(x, 0, x, height, fill='lightgray')
            self.canvas.create_text(x, height - 10, text=str(i), anchor=tk.N)

            y = i * height / 100
            self.canvas.create_line(0, y, width, y, fill='lightgray')
            self.canvas.create_text(10, y, text=str(i), anchor=tk.W)
        depot_x = 0
        depot_y = height
        self.canvas.create_oval(depot_x - 5, depot_y - 5, depot_x + 5, depot_y + 5, fill='red', outline='black')
        self.canvas.create_text(depot_x + 10, depot_y - 10, text="Depot", anchor=tk.W)

        if self.current_solution:
            self.draw_solution()

    def draw_solution(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']

        for i, vehicle in enumerate(self.current_solution):
            color = colors[i % len(colors)]
            prev_x, prev_y = 0, height

            for package in vehicle.route:
                pkg_x = package.x * width / 100
                pkg_y = height - (package.y * height / 100)
                self.canvas.create_line(prev_x, prev_y, pkg_x, pkg_y, fill=color, width=2)
                self.canvas.create_oval(pkg_x - 4, pkg_y - 4, pkg_x + 4, pkg_y + 4,
                                        fill=self.get_priority_color(package.priority), outline='black')

                # Package label
                self.canvas.create_text(pkg_x + 8, pkg_y - 8,
                                        text=f"P{package.id}({package.weight}kg)",
                                        anchor=tk.W, fill=color)

                prev_x, prev_y = pkg_x, pkg_y
            self.canvas.create_line(prev_x, prev_y, 0, height, fill=color, width=2, dash=(2, 2))

            self.canvas.create_text(width - 10, 20 + i * 20,
                                    text=f"Vehicle {vehicle.id} ({sum(p.weight for p in vehicle.packages)}/{vehicle.capacity}kg)",
                                    anchor=tk.E, fill=color)

    def get_priority_color(self, priority):
        colors = {
            1: 'red',
            2: 'orange',
            3: 'yellow',
            4: 'lightgreen',
            5: 'green'
        }
        return colors.get(priority, 'gray')

    def create_results_panel(self, parent):
        # Results Frame with Statistics
        results_frame = tk.LabelFrame(parent, text="Results Statistics", padx=5, pady=5)
        results_frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # Notebook for multiple tabs
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)

        # Summary Tab
        summary_tab = tk.Frame(self.results_notebook)
        self.results_notebook.add(summary_tab, text="Summary")

        self.summary_text = tk.Text(summary_tab, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # Detailed Routes Tab
        routes_tab = tk.Frame(self.results_notebook)
        self.results_notebook.add(routes_tab, text="Detailed Routes")

        self.routes_tree = ttk.Treeview(routes_tab, columns=('vehicle', 'stop', 'package', 'distance'), show='headings')
        self.routes_tree.heading('vehicle', text='Vehicle')
        self.routes_tree.heading('stop', text='Stop')
        self.routes_tree.heading('package', text='Package')
        self.routes_tree.heading('distance', text='Distance (km)')

        self.routes_tree.column('vehicle', width=80)
        self.routes_tree.column('stop', width=80)
        self.routes_tree.column('package', width=150)
        self.routes_tree.column('distance', width=100)

        scrollbar = ttk.Scrollbar(routes_tab, orient=tk.VERTICAL, command=self.routes_tree.yview)
        self.routes_tree.configure(yscrollcommand=scrollbar.set)

        self.routes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def update_algorithm_params(self):

        for widget in self.params_frame.winfo_children():
            if widget not in [self.sa_cooling_label, self.sa_cooling_slider,
                              self.ga_population_label, self.ga_population_entry,
                              self.ga_mutation_label, self.ga_mutation_entry]:
                widget.pack_forget()


        if self.algorithm_var.get() == "simulated_annealing":
            self.sa_cooling_label.pack(anchor=tk.W)
            self.sa_cooling_slider.pack(fill=tk.X)

            self.ga_population_label.pack_forget()
            self.ga_population_entry.pack_forget()
            self.ga_mutation_label.pack_forget()
            self.ga_mutation_entry.pack_forget()
        else:
            self.ga_population_label.pack(anchor=tk.W)
            self.ga_population_entry.pack(fill=tk.X)
            self.ga_mutation_label.pack(anchor=tk.W)
            self.ga_mutation_entry.pack(fill=tk.X)
            self.sa_cooling_label.pack_forget()
            self.sa_cooling_slider.pack_forget()

    def add_vehicle(self):

        try:
            capacity = float(self.vehicle_capacity_entry.get())


            if capacity <= 0:
                raise ValueError("Capacity must be greater than 0 kg")
            if capacity > 100:
                raise ValueError("Capacity cannot exceed 100 kg")


            vehicle_id = len(self.vehicles)
            self.vehicles.append(Vehicle(vehicle_id, capacity))
            self.vehicle_tree.insert('', 'end', values=(vehicle_id, f"{capacity:.1f}"))

            # Clear and reset capacity entry
            self.vehicle_capacity_entry.delete(0, tk.END)
            self.vehicle_capacity_entry.insert(0, "100")  # Reset to max allowed value

        except ValueError as e:
            messagebox.showerror("Invalid Capacity", str(e))
            self.vehicle_capacity_entry.delete(0, tk.END)
            self.vehicle_capacity_entry.insert(0, "100")  # Reset on error
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add vehicle: {str(e)}")

    def remove_vehicle(self):
        selected = self.vehicle_tree.selection()
        if not selected:
            return

        item = selected[0]
        vehicle_id = int(self.vehicle_tree.item(item, 'values')[0])

        self.vehicle_tree.delete(item)

        self.vehicles = [v for v in self.vehicles if v.id != vehicle_id]

        for i, v in enumerate(self.vehicles):
            v.id = i
        self.vehicle_tree.delete(*self.vehicle_tree.get_children())
        for v in self.vehicles:
            self.vehicle_tree.insert('', 'end', values=(v.id, v.capacity))

    def add_package(self):
        """Add new package with strict validation"""
        try:
            # Validate inputs
            x = float(self.pkg_x_entry.get())
            y = float(self.pkg_y_entry.get())
            weight = float(self.pkg_weight_entry.get())
            priority = int(self.pkg_priority_var.get())

            if not (0 <= x <= 100) or not (0 <= y <= 100):
                raise ValueError("Coordinates must be 0-100")
            if weight <= 0 or weight > 100:
                raise ValueError("Weight must be 0-100 kg")
            if priority not in range(1, 6):
                raise ValueError("Priority must be 1-5")

            # Add package
            pkg_id = len(self.packages)
            self.packages.append(Package(pkg_id, x, y, weight, priority))
            self.package_tree.insert('', 'end',
                                     values=(pkg_id, f"{x:.1f}", f"{y:.1f}",
                                             f"{weight:.1f}", priority))

            # Clear inputs
            self.pkg_x_entry.delete(0, tk.END)
            self.pkg_y_entry.delete(0, tk.END)
            self.pkg_weight_entry.delete(0, tk.END)

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))


    def remove_package(self):
        selected = self.package_tree.selection()
        if not selected:
            return

        item = selected[0]
        pkg_id = int(self.package_tree.item(item, 'values')[0])

        # Remove from treeview
        self.package_tree.delete(item)

        # Remove from packages list
        self.packages = [p for p in self.packages if p.id != pkg_id]

        # Reassign IDs to maintain order
        for i, p in enumerate(self.packages):
            p.id = i

        # Refresh treeview
        self.package_tree.delete(*self.package_tree.get_children())
        for p in self.packages:
            self.package_tree.insert('', 'end', values=(p.id, f"{p.x:.1f}", f"{p.y:.1f}",
                                                        f"{p.weight:.1f}", p.priority))

    def clear_packages(self):
        if messagebox.askyesno("Confirm", "Clear all packages?"):
            self.packages = []
            self.package_tree.delete(*self.package_tree.get_children())

    def generate_random_packages(self):
        if not self.vehicles:
            messagebox.showerror("Error", "Please add vehicles first")
            return
        total_capacity = sum(v.capacity for v in self.vehicles)
        num_packages = random.randint(3, int(total_capacity / 10))

        self.clear_packages()

        for i in range(num_packages):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            weight = random.uniform(1, min(50, total_capacity / 5))  # Reasonable weight
            priority = random.randint(1, 5)

            self.packages.append(Package(i, x, y, weight, priority))
            self.package_tree.insert('', 'end', values=(i, f"{x:.1f}", f"{y:.1f}",
                                                        f"{weight:.1f}", priority))

    def run_optimization(self):
        if not self.vehicles:
            messagebox.showerror("Error", "Please add at least one vehicle")
            return

        if not self.packages:
            messagebox.showerror("Error", "Please add at least one package")
            return

        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress['value'] = 0
        self.root.update()

        # Get algorithm parameters
        if self.algorithm_var.get() == "simulated_annealing":
            cooling_rate = self.sa_cooling_slider.get() / 100
            solution = simulated_annealing(
                self.packages, self.vehicles,
                cooling_rate=cooling_rate,
                progress_callback=self.update_progress
            )
        else:
            try:
                population_size = int(self.ga_population_entry.get())
                mutation_rate = float(self.ga_mutation_entry.get())

                if not (50 <= population_size <= 100):
                    raise ValueError("Population size must be between 50 and 100")
                if not (0.01 <= mutation_rate <= 0.1):
                    raise ValueError("Mutation rate must be between 0.01 and 0.1")

            except ValueError as e:
                messagebox.showerror("Error", f"Invalid parameters: {str(e)}")
                self.stop_optimization()
                return

            solution = genetic_algorithm(
                self.packages, self.vehicles,
                population_size=population_size,
                mutation_rate=mutation_rate,
                progress_callback=self.update_progress
            )

        if self.running:  # Only update if not stopped
            self.current_solution = solution
            self.display_results(solution)

        self.stop_optimization()

    def stop_optimization(self):
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def reset_solution(self):
        self.current_solution = None
        self.summary_text.delete(1.0, tk.END)
        self.routes_tree.delete(*self.routes_tree.get_children())
        self.draw_grid()

    def update_progress(self, value):
        if self.running:
            self.progress['value'] = value
            self.root.update()

    def display_results(self, solution):
        # Calculate statistics
        total_distance = calculate_total_distance(solution)
        num_packages = len(self.packages)
        delivered_packages = sum(len(v.packages) for v in solution)
        capacity_utilization = sum(sum(p.weight for p in v.packages) for v in solution) / sum(
            v.capacity for v in solution) * 100

        # Priority distribution
        priority_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for v in solution:
            for p in v.packages:
                priority_counts[p.priority] += 1

        # Update summary tab
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END,
                                 f"Optimization Results ({'Simulated Annealing' if self.algorithm_var.get() == 'simulated_annealing' else 'Genetic Algorithm'})\n\n")
        self.summary_text.insert(tk.END, f"Total distance traveled: {total_distance:.2f} km\n")
        self.summary_text.insert(tk.END, f"Packages delivered: {delivered_packages}/{num_packages}\n")
        self.summary_text.insert(tk.END, f"Capacity utilization: {capacity_utilization:.1f}%\n\n")

        self.summary_text.insert(tk.END, "Priority distribution of delivered packages:\n")
        for prio, count in priority_counts.items():
            self.summary_text.insert(tk.END, f"  Priority {prio}: {count} packages\n")

        # Update detailed routes tab
        self.routes_tree.delete(*self.routes_tree.get_children())
        for vehicle in solution:
            prev_x, prev_y = 0, 0  # Depot
            segment_distance = 0
            stop_num = 1

            # Depot start
            self.routes_tree.insert('', 'end', values=(f"Vehicle {vehicle.id}", "Depot", "", "0.00"))

            for package in vehicle.route:
                distance = euclidean_distance(prev_x, prev_y, package.x, package.y)
                segment_distance += distance

                self.routes_tree.insert('', 'end',
                                        values=(f"Vehicle {vehicle.id}",
                                                f"Stop {stop_num}",
                                                f"P{package.id} ({package.x:.1f},{package.y:.1f}) {package.weight}kg prio:{package.priority}",
                                                f"{segment_distance:.2f}"))

                prev_x, prev_y = package.x, package.y
                stop_num += 1

            # Return to depot
            distance = euclidean_distance(prev_x, prev_y, 0, 0)
            segment_distance += distance
            self.routes_tree.insert('', 'end',
                                    values=(f"Vehicle {vehicle.id}",
                                            "Depot (return)",
                                            "",
                                            f"{segment_distance:.2f}"))
        self.draw_grid()
if __name__ == "__main__":
    root = tk.Tk()
    app = DeliveryApp(root)
    root.mainloop()