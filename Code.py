import random
import pandas as pd

# --------------------------
# Helper: Convert string to boolean
# --------------------------
def to_bool(val):
    return str(val).strip().lower() in ['true', '1']

# --------------------------
# Load Shelves from shelve.txt
# --------------------------
def load_shelves(filename):
    df = pd.read_csv(filename)
    shelves = {}
    for _, row in df.iterrows():
        shelf_id = row['ShelfID']
        shelves[shelf_id] = {
            'name': row['Name'],
            'capacity': row['Capacity'],
            'type': row['Type'],
            'secured': to_bool(row['Secured']),
            'visibility': row['Visibility'].strip()
        }
    return shelves

# --------------------------
# Load Products from product.txt
# --------------------------
def load_products(filename):
    df = pd.read_csv(filename)
    products = {}
    for _, row in df.iterrows():
        prod_id = row['ProductID']
        products[prod_id] = {
            'name': row['Name'],
            'weight': row['Weight'],
            'category': row['Category'],
            'high_demand': to_bool(row['HighDemand']),
            'perishable': to_bool(row['Perishable']),
            'bulky': to_bool(row['Bulky']),
            'hazardous': to_bool(row['Hazardous']),
            'refrigerated': to_bool(row['Refrigerated']),
            'promotional': to_bool(row['Promotional']),
            'expensive': to_bool(row['Expensive'])
        }
    return products

# Load shelves and products from the provided files.
shelves = load_shelves("D:\\Semester 6\\AI\\Assignment2\\shelves.txt")
products = load_products("D:\\Semester 6\\AI\\Assignment2\\products.txt")
product_keys = list(products.keys())

# --------------------------
# GA Parameters
# --------------------------
POP_SIZE = 50         # Number of solutions in the population
GENERATIONS = 250     # Number of generations to run
MUTATION_RATE = 0.2   # Probability of mutation per gene

# --------------------------
# Fitness Function: Check All Constraints
# --------------------------
def fitness(chromosome):
    penalty = 0
    shelf_usage = {shelf: 0 for shelf in shelves.keys()}
    shelf_products = {shelf: [] for shelf in shelves.keys()}
    
    # Build shelf usage and track product placements.
    for i, shelf_assignment in enumerate(chromosome):
        prod_id = product_keys[i]
        prod = products[prod_id]
        shelf_usage[shelf_assignment] += prod['weight']
        shelf_products[shelf_assignment].append(prod_id)
    
    # 1. Shelf Capacity & Weight Limit
    for shelf_id, usage in shelf_usage.items():
        capacity = shelves[shelf_id]['capacity']
        if usage > capacity:
            penalty += (usage - capacity) * 10  # heavy penalty for overloading
    
    # 2. High-Demand Product Accessibility
    for i, shelf_assignment in enumerate(chromosome):
        prod_id = product_keys[i]
        prod = products[prod_id]
        if prod.get('high_demand', False):
            if shelves[shelf_assignment]['type'] not in ['accessible', 'high_visibility']:
                penalty += 5
    
    # 3. Product Category Segmentation (penalty if same category split)
    category_shelf = {}
    for i, shelf_assignment in enumerate(chromosome):
        prod_id = product_keys[i]
        prod = products[prod_id]
        cat = prod.get('category', None)
        if cat:
            if cat in category_shelf and category_shelf[cat] != shelf_assignment:
                penalty += 1
            else:
                category_shelf[cat] = shelf_assignment

    # 4. Perishable vs. Non-Perishable Separation
    for i, shelf_assignment in enumerate(chromosome):
        prod_id = product_keys[i]
        prod = products[prod_id]
        if prod.get('perishable', False):
            if shelves[shelf_assignment]['type'] != 'refrigerated':
                penalty += 10
    
    # 5. Hazardous and Allergen-Free Zones
    for i, shelf_assignment in enumerate(chromosome):
        prod_id = product_keys[i]
        prod = products[prod_id]
        if prod.get('hazardous', False):
            if shelves[shelf_assignment]['type'] != 'hazardous':
                penalty += 10

    # 6. Product Compatibility and Cross-Selling (e.g., Pasta & Pasta Sauce should be together)
    if 'P5' in product_keys and 'P6' in product_keys:
        if chromosome[product_keys.index('P5')] != chromosome[product_keys.index('P6')]:
            penalty += 5

    # 7. Restocking Efficiency: Bulky items should be on lower shelves.
    for i, shelf_assignment in enumerate(chromosome):
        prod_id = product_keys[i]
        prod = products[prod_id]
        if prod.get('bulky', False):
            if shelves[shelf_assignment]['type'] != 'lower':
                penalty += 5

    # 8. Refrigeration Efficiency: Already covered with perishable items.
    # (Assuming perishable items require refrigerated zones.)
    
    # 9. Promotional and Discounted Items Visibility
    for i, shelf_assignment in enumerate(chromosome):
        prod_id = product_keys[i]
        prod = products[prod_id]
        if prod.get('promotional', False):
            if shelves[shelf_assignment]['visibility'].lower() != 'high':
                penalty += 5

    # 10. Theft Prevention: Expensive items must be placed on secured shelves.
    for i, shelf_assignment in enumerate(chromosome):
        prod_id = product_keys[i]
        prod = products[prod_id]
        if prod.get('expensive', False):
            if not shelves[shelf_assignment].get('secured', False):
                penalty += 10

    return penalty

# --------------------------
# GA Operators: Initialization, Selection, Crossover, Mutation
# --------------------------
def create_chromosome():
    """Randomly assign each product to one of the available shelves."""
    return [random.choice(list(shelves.keys())) for _ in range(len(product_keys))]

def initial_population():
    return [create_chromosome() for _ in range(POP_SIZE)]

def selection(pop):
    """Tournament selection: pick two random chromosomes and select the one with lower penalty."""
    a = random.choice(pop)
    b = random.choice(pop)
    return a if fitness(a) < fitness(b) else b

def crossover(parent1, parent2):
    """Always perform one-point crossover."""
    point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome):
    """Mutate by randomly reassigning a shelf based on MUTATION_RATE."""
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = random.choice(list(shelves.keys()))
    return chromosome

# --------------------------
# Genetic Algorithm Main Loop
# --------------------------
def genetic_algorithm():
    population = initial_population()
    best_chromosome = None
    best_fit = float('inf')

    # for _ in range(POP_SIZE):
    #     print(population[_])
    
    for gen in range(GENERATIONS):
        new_population = []
        while len(new_population) < POP_SIZE:
            parent1 = selection(population)
            parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = new_population[:POP_SIZE]
        
        # Track best solution in current generation.
        for chrom in population:
            current_fit = fitness(chrom)
            if current_fit < best_fit:
                best_fit = current_fit
                best_chromosome = chrom

        # if gen % 10 == 0:
        #     print(f"Generation {gen}: Best Fitness = {best_fit}")
    
    return best_chromosome, best_fit

# --------------------------
# Save the Best Solution to Excel
# --------------------------
def save_to_excel(chromosome, filename="optimized_shelf_allocation.xlsx"):
    records = []
    for i, shelf_assignment in enumerate(chromosome):
        prod_id = product_keys[i]
        prod = products[prod_id]
        shelf_info = shelves[shelf_assignment]
        record = {
            "Product ID": prod_id,
            "Product Name": prod['name'],
            "Weight": prod['weight'],
            "Category": prod.get('category', ''),
            "Assigned Shelf": shelf_assignment,
            "Shelf Name": shelf_info['name'],
            "Shelf Capacity": shelf_info['capacity'],
            "Shelf Type": shelf_info['type'],
            "Shelf Secured": shelf_info['secured'],
            "Shelf Visibility": shelf_info['visibility']
        }
        records.append(record)
    df = pd.DataFrame(records)
    df.to_excel(filename, index=False)
    print(f"Optimized shelf allocation saved to {filename}")

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    best_solution, best_fit = genetic_algorithm()
    print("\nBest solution found:")
    for i, shelf_assignment in enumerate(best_solution):
        prod_id = product_keys[i]
        print(f"{prod_id} ({products[prod_id]['name']}) -> Shelf {shelf_assignment} ({shelves[shelf_assignment]['name']})")
    print(f"\nBest Fitness (Total Penalty): {best_fit}")
    
    # Save the best solution to an Excel file.
    save_to_excel(best_solution)
