import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, constants
import datetime
import json

class EcologicalMagneticFieldSimulator:
    def __init__(self, region_size=100, time_steps=1000):
        self.region_size = region_size
        self.time_steps = time_steps
        self.results = []
        self.log = []
        
        # Constantes fundamentais
        self.mu0 = constants.mu_0
        self.epsilon0 = constants.epsilon_0
        self.hbar = constants.hbar
        
        # Constantes hipotéticas do CME
        self.alpha_bio = 1.6e-19  # Constante de acoplamento biológico (C·s/kg)
        self.beta_ent = 2.4e-12   # Constante de acoplamento entrópico (J/K·T·m)
        self.gamma_quant = 3.8e-8 # Constante de acoplamento quântico (m²/s·T)
        
        # Inicializar grades espaciais
        self.x = np.linspace(0, 10, region_size)
        self.y = np.linspace(0, 10, region_size)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        
        # Parâmetros iniciais
        self.bio_density = np.zeros((region_size, region_size))
        self.entropy_density = np.zeros((region_size, region_size))
        self.cme_field = np.zeros((region_size, region_size))
        
        self.log_event("INIT", "Simulador de Campo Magnético Ecológico inicializado")
    
    def log_event(self, event_type, message, data=None):
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event": event_type,
            "message": message,
            "data": data
        }
        self.log.append(log_entry)
        print(f"[{timestamp}] {event_type}: {message}")
    
    def initialize_conditions(self, bio_pattern="forest", entropy_pattern="gradient"):
        """Inicializa condições biológicas e entrópicas"""
        
        if bio_pattern == "forest":
            # Padrão de densidade biológica semelhante a uma floresta
            self.bio_density = np.exp(-((self.xx-5)**2 + (self.yy-5)**2)/8) * 0.7
            self.bio_density += 0.3 * np.random.random((self.region_size, self.region_size))
        
        if entropy_pattern == "gradient":
            # Gradiente de entropia do canto inferior esquerdo (baixa entropia)
            # para o canto superior direito (alta entropia)
            self.entropy_density = (self.xx + self.yy) / 20
            self.entropy_density += 0.1 * np.random.random((self.region_size, self.region_size))
        
        self.log_event("INIT_COND", "Condições iniciais estabelecidas", {
            "bio_pattern": bio_pattern,
            "entropy_pattern": entropy_pattern
        })
    
    def calculate_cme_field(self, time_step):
        """Calcula o campo magnético ecológico"""
        
        # Calcular gradiente de densidade biológica
        bio_grad_x, bio_grad_y = np.gradient(self.bio_density)
        bio_grad_magnitude = np.sqrt(bio_grad_x**2 + bio_grad_y**2)
        
        # Calcular gradiente de entropia
        entropy_grad_x, entropy_grad_y = np.gradient(self.entropy_density)
        entropy_grad_magnitude = np.sqrt(entropy_grad_x**2 + entropy_grad_y**2)
        
        # Equação principal do CME (forma simplificada)
        self.cme_field = (self.alpha_bio * bio_grad_magnitude - 
                         self.beta_ent * entropy_grad_magnitude * np.sin(0.1 * time_step) +
                         self.gamma_quant * np.random.normal(0, 0.1, (self.region_size, self.region_size)))
        
        # Aplicar filtro de suavização
        from scipy.ndimage import gaussian_filter
        self.cme_field = gaussian_filter(self.cme_field, sigma=1)
        
        return self.cme_field
    
    def update_system(self, time_step):
        """Atualiza o estado do sistema"""
        
        # Atualizar densidade biológica (crescimento logístico)
        carrying_capacity = 1.0
        growth_rate = 0.01 * (1 + 0.1 * self.cme_field)
        self.bio_density += growth_rate * self.bio_density * (1 - self.bio_density / carrying_capacity)
        
        # Atualizar entropia (segunda lei com contribuição do CME)
        entropy_production = 0.001 * np.ones_like(self.entropy_density)
        entropy_reduction = 0.0005 * self.cme_field**2
        entropy_reduction = np.clip(entropy_reduction, 0, 0.01)  # Limitar redução
        
        self.entropy_density += entropy_production - entropy_reduction
        self.entropy_density = np.clip(self.entropy_density, 0, 1)  # Manter entre 0 e 1
        
        # Registrar resultados
        result = {
            "time_step": time_step,
            "total_bio_mass": np.sum(self.bio_density),
            "avg_entropy": np.mean(self.entropy_density),
            "max_cme_strength": np.max(self.cme_field),
            "min_cme_strength": np.min(self.cme_field),
            "cme_variance": np.var(self.cme_field),
            "bio_entropy_correlation": np.corrcoef(self.bio_density.flatten(), 
                                                  self.entropy_density.flatten())[0, 1]
        }
        
        self.results.append(result)
        
        # Registrar eventos significativos
        if time_step % 100 == 0:
            self.log_event("UPDATE", f"Sistema atualizado no passo de tempo {time_step}", result)
        
        return result
    
    def run_simulation(self):
        """Executa a simulação completa"""
        self.log_event("SIM_START", "Iniciando simulação do CME")
        
        self.initialize_conditions()
        
        for t in range(self.time_steps):
            self.calculate_cme_field(t)
            self.update_system(t)
        
        self.log_event("SIM_END", "Simulação do CME concluída", {
            "total_time_steps": self.time_steps,
            "final_bio_mass": self.results[-1]["total_bio_mass"],
            "final_avg_entropy": self.results[-1]["avg_entropy"]
        })
        
        return self.results
    
    def visualize_results(self):
        """Visualiza os resultados da simulação"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Campo CME
        im1 = axes[0, 0].imshow(self.cme_field, cmap='viridis', origin='lower')
        axes[0, 0].set_title('Campo Magnético Ecológico')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Densidade biológica
        im2 = axes[0, 1].imshow(self.bio_density, cmap='Greens', origin='lower')
        axes[0, 1].set_title('Densidade Biológica')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Densidade de entropia
        im3 = axes[0, 2].imshow(self.entropy_density, cmap='hot', origin='lower')
        axes[0, 2].set_title('Densidade de Entropia')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Série temporal de biomassa
        time_steps = [r["time_step"] for r in self.results]
        bio_mass = [r["total_bio_mass"] for r in self.results]
        axes[1, 0].plot(time_steps, bio_mass)
        axes[1, 0].set_title('Biomassa Total ao Longo do Tempo')
        axes[1, 0].set_xlabel('Passo de Tempo')
        axes[1, 0].set_ylabel('Biomassa')
        
        # Série temporal de entropia
        entropy = [r["avg_entropy"] for r in self.results]
        axes[1, 1].plot(time_steps, entropy)
        axes[1, 1].set_title('Entropia Média ao Longo do Tempo')
        axes[1, 1].set_xlabel('Passo de Tempo')
        axes[1, 1].set_ylabel('Entropia')
        
        # Força do CME vs. Biomassa
        cme_strength = [r["max_cme_strength"] for r in self.results]
        axes[1, 2].scatter(bio_mass, cme_strength, alpha=0.5)
        axes[1, 2].set_title('Relação entre Biomassa e Força do CME')
        axes[1, 2].set_xlabel('Biomassa')
        axes[1, 2].set_ylabel('Força Máxima do CME')
        
        plt.tight_layout()
        plt.savefig('cme_simulation_results.png')
        plt.close()
        
        self.log_event("VISUALIZATION", "Visualizações geradas e salvas")
    
    def generate_report(self):
        """Gera um relatório detalhado da simulação"""
        report = {
            "simulation_parameters": {
                "region_size": self.region_size,
                "time_steps": self.time_steps,
                "alpha_bio": self.alpha_bio,
                "beta_ent": self.beta_ent,
                "gamma_quant": self.gamma_quant
            },
            "final_results": self.results[-1] if self.results else None,
            "summary_statistics": self.calculate_summary_statistics(),
            "mythological_connections": self.analyze_mythological_connections(),
            "log_entries": self.log
        }
        
        with open('cme_simulation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log_event("REPORT", "Relatório detalhado gerado")
        
        return report
    
    def calculate_summary_statistics(self):
        """Calcula estatísticas resumidas da simulação"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        return {
            "total_bio_mass_growth": df["total_bio_mass"].iloc[-1] - df["total_bio_mass"].iloc[0],
            "entropy_change": df["avg_entropy"].iloc[-1] - df["avg_entropy"].iloc[0],
            "avg_cme_strength": df["max_cme_strength"].mean(),
            "bio_entropy_correlation_avg": df["bio_entropy_correlation"].mean(),
            "cme_stability": df["max_cme_strength"].std() / df["max_cme_strength"].mean() if df["max_cme_strength"].mean() != 0 else 0
        }
    
    def analyze_mythological_connections(self):
        """Analisa conexões com mitologias antigas baseadas nos resultados"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        avg_bio_entropy_corr = df["bio_entropy_correlation"].mean()
        cme_stability = df["max_cme_strength"].std() / df["max_cme_strength"].mean() if df["max_cme_strength"].mean() != 0 else 0
        
        connections = {
            "chinese_qi": {
                "strength": min(1.0, max(0, (1 - cme_stability) * avg_bio_entropy_corr)),
                "interpretation": "O fluxo de Qi está associado à estabilidade do CME e à correlação entre vida e entropia"
            },
            "egyptian_maat": {
                "strength": min(1.0, max(0, (1 - abs(avg_bio_entropy_corr)))),
                "interpretation": "Maat representa equilíbrio; valor mais alto quando vida e entropia estão em equilíbrio"
            },
            "norse_wyrd": {
                "strength": min(1.0, max(0, cme_stability)),
                "interpretation": "Wyrd como tecido do destino é mais perceptível quando o CME é estável"
            },
            "greek_pneuma": {
                "strength": min(1.0, max(0, df["max_cme_strength"].mean())),
                "interpretation": "Pneuma como sopro vital correlaciona-se com a força média do CME"
            }
        }
        
        return connections

# Executar a simulação
if __name__ == "__main__":
    simulator = EcologicalMagneticFieldSimulator(region_size=50, time_steps=500)
    results = simulator.run_simulation()
    simulator.visualize_results()
    report = simulator.generate_report()
    
    print("Simulação concluída. Relatório salvo em 'cme_simulation_report.json'")
    print("Visualizações salvas em 'cme_simulation_results.png'")
